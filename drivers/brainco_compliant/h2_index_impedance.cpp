#include "stark-sdk.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace
{

constexpr uint32_t BAUDRATE = 460800;
constexpr uint8_t RIGHT_HAND_ID = 0x7f;
constexpr std::size_t MOTOR_COUNT = 6;
constexpr std::size_t INDEX = 2;

constexpr uint16_t PREPOSITION = 50;
constexpr uint16_t PREPOSITION_SPEED = 80;
constexpr uint16_t TARGET_DELTA = 340;
constexpr uint16_t MIN_IMPEDANCE_TRAVEL = 20;
constexpr uint16_t MAX_INDEX_POSITION = 450;
constexpr uint16_t OTHER_POSITION_DELTA_STOP = 30;
constexpr int16_t OTHER_SPEED_STOP = 100;
constexpr int16_t SPEED_STOP = 850;
constexpr int16_t CURRENT_COMMAND_LIMIT = 200;
constexpr int16_t CURRENT_FEEDBACK_LIMIT = 260;
constexpr int16_t ZERO_CURRENT_TOLERANCE = 10;
constexpr int16_t ZERO_SPEED_TOLERANCE = 30;

constexpr double KP = 5.0;
constexpr double KD = 0.05;
constexpr double RAMP_RATE = 0.04;
constexpr double EMA_ALPHA = 0.30;
constexpr double CLOSING_SPEED_SOFT_LIMIT = 0.25;
constexpr double CLOSING_SPEED_ZERO_CURRENT = 0.55;

constexpr auto CONTROL_PERIOD = std::chrono::milliseconds(20);
constexpr auto STATUS_PERIOD = std::chrono::milliseconds(50);
constexpr auto BASELINE_DURATION = std::chrono::seconds(1);
constexpr auto POSITION_TIMEOUT = std::chrono::seconds(5);
constexpr auto ZERO_TIMEOUT = std::chrono::seconds(1);
constexpr auto ARMED_DURATION = std::chrono::milliseconds(500);
constexpr auto RAMP_DURATION = std::chrono::seconds(10);
constexpr auto HOLD_DURATION = std::chrono::seconds(5);
constexpr auto CONTROL_DEADLINE = std::chrono::milliseconds(50);

constexpr std::array<const char*, MOTOR_COUNT> MOTOR_NAMES = {
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
};

std::atomic<bool> stop_requested{false};

void signal_handler(int)
{
    stop_requested.store(true);
}

struct Options
{
    bool confirmed{false};
    std::string port;
    std::string expected_serial;
    std::string output_prefix;
    std::string h0_gate_meta;
};

void print_usage(const char* executable)
{
    std::cout
        << "Usage:\n  " << executable
        << " --confirm-impedance-test"
        << " --port <device>"
        << " --expected-serial <serial>"
        << " --h0-gate-meta <path>"
        << " --output-prefix <path>\n";
}

std::optional<Options> parse_options(int argc, char** argv)
{
    Options options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return std::nullopt;
        }

        if (arg == "--confirm-impedance-test")
        {
            options.confirmed = true;
            continue;
        }

        auto read_value = [&](std::string& target) -> bool
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value after " << arg << "\n";
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "--port")
        {
            if (!read_value(options.port))
                return std::nullopt;
            continue;
        }

        if (arg == "--expected-serial")
        {
            if (!read_value(options.expected_serial))
                return std::nullopt;
            continue;
        }

        if (arg == "--output-prefix")
        {
            if (!read_value(options.output_prefix))
                return std::nullopt;
            continue;
        }

        if (arg == "--h0-gate-meta")
        {
            if (!read_value(options.h0_gate_meta))
                return std::nullopt;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return std::nullopt;
    }

    if (
        !options.confirmed
        || options.port.empty()
        || options.expected_serial.empty()
        || options.h0_gate_meta.empty()
        || options.output_prefix.empty()
    )
    {
        print_usage(argv[0]);
        return std::nullopt;
    }

    return options;
}

std::unordered_map<std::string, std::string> read_meta(
    const std::string& path
)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Failed to open H0 gate meta file");

    std::unordered_map<std::string, std::string> values;
    std::string line;

    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            continue;
        values[line.substr(0, separator)] = line.substr(separator + 1);
    }

    return values;
}

bool h0_gate_passed(
    const std::unordered_map<std::string, std::string>& meta,
    const std::string& serial,
    const std::string& firmware,
    uint16_t live_max_current,
    uint16_t live_protected_current
)
{
    auto equals = [&](const std::string& key, const std::string& expected)
    {
        const auto it = meta.find(key);
        return it != meta.end() && it->second == expected;
    };

    const auto close_current_it = meta.find("close_current");
    const auto visible_target_it = meta.find("visible_target");
    const auto max_current_it = meta.find("index_max_current_mA");
    const auto protected_current_it = meta.find(
        "index_protected_current_mA"
    );

    if (
        close_current_it == meta.end()
        || visible_target_it == meta.end()
        || max_current_it == meta.end()
        || protected_current_it == meta.end()
    )
        return false;

    int close_current = 0;
    int visible_target = 0;
    int h0_max_current = 0;
    int h0_protected_current = 0;

    try
    {
        auto parse_exact_int = [](const std::string& text)
        {
            std::size_t consumed = 0;
            const int value = std::stoi(text, &consumed);
            if (consumed != text.size())
                throw std::invalid_argument("trailing characters");
            return value;
        };

        close_current = parse_exact_int(close_current_it->second);
        visible_target = parse_exact_int(visible_target_it->second);
        h0_max_current = parse_exact_int(max_current_it->second);
        h0_protected_current =
            parse_exact_int(protected_current_it->second);
    }
    catch (const std::exception&)
    {
        return false;
    }

    return equals("test", "h0_index_current_visible_v1")
        && equals("serial", serial)
        && equals("firmware", firmware)
        && equals("hardware", "5")
        && equals("sku", "1")
        && equals("result", "PASS")
        && equals("current_target_reached", "1")
        && equals("returned_open", "1")
        && equals("zero_confirmed", "1")
        && equals("final_position", "0")
        && close_current > 0
        && close_current <= CURRENT_COMMAND_LIMIT
        && visible_target >= 200
        && h0_max_current == static_cast<int>(live_max_current)
        && h0_protected_current
            == static_cast<int>(live_protected_current);
}

std::string safe_text(const char* value)
{
    return value ? std::string(value) : std::string();
}

CDeviceInfo* read_device_info_with_retry(DeviceHandler* handle)
{
    for (int attempt = 0; attempt < 5; ++attempt)
    {
        if (CDeviceInfo* info = stark_get_device_info(handle, RIGHT_HAND_ID))
            return info;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return nullptr;
}

std::string wall_time_iso8601()
{
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream stream;
    stream << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return stream.str();
}

bool is_right_sku(SkuType sku)
{
    return sku == SkuType::SKU_TYPE_SMALL_RIGHT
        || sku == SkuType::SKU_TYPE_MEDIUM_RIGHT;
}

bool is_revo2(StarkHardwareType hardware)
{
    return hardware == StarkHardwareType::STARK_HARDWARE_TYPE_REVO2_BASIC
        || hardware == StarkHardwareType::STARK_HARDWARE_TYPE_REVO2_TOUCH
        || hardware
            == StarkHardwareType::STARK_HARDWARE_TYPE_REVO2_TOUCH_PRESSURE;
}

bool unsafe_state(uint8_t state)
{
    return state == MOTOR_STATE_TURBO
        || state == MOTOR_STATE_UNKNOWN;
}

struct Sample
{
    std::array<uint16_t, MOTOR_COUNT> positions{};
    std::array<int16_t, MOTOR_COUNT> speeds{};
    std::array<int16_t, MOTOR_COUNT> currents{};
    std::array<uint8_t, MOTOR_COUNT> states{};
    long long read_latency_us{0};
};

bool read_sample(DeviceHandler* handle, Sample& sample)
{
    const auto start = std::chrono::steady_clock::now();
    CMotorStatusData* status =
        stark_get_motor_status(handle, RIGHT_HAND_ID);
    const auto finish = std::chrono::steady_clock::now();

    sample.read_latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            finish - start
        ).count();

    if (!status)
        return false;

    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
    {
        sample.positions[i] = status->positions[i];
        sample.speeds[i] = status->speeds[i];
        sample.currents[i] = status->currents[i];
        sample.states[i] = status->states[i];
    }

    free_motor_status_data(status);
    return true;
}

class ProcessLock
{
public:
    bool acquire()
    {
#ifdef __linux__
        fd_ = ::open(
            "/tmp/brainco_h0_right.lock",
            O_CREAT | O_RDWR,
            0660
        );
        if (fd_ < 0 || ::flock(fd_, LOCK_EX | LOCK_NB) != 0)
            return false;
#endif
        return true;
    }

    ~ProcessLock()
    {
#ifdef __linux__
        if (fd_ >= 0)
        {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
#endif
    }

private:
#ifdef __linux__
    int fd_{-1};
#endif
};

class DeviceGuard
{
public:
    explicit DeviceGuard(DeviceHandler* handle)
        : handle_(handle)
    {
    }

    ~DeviceGuard()
    {
        if (handle_)
            modbus_close(handle_);
    }

private:
    DeviceHandler* handle_;
};

void send_index_current(DeviceHandler* handle, int16_t current)
{
    stark_set_finger_current(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX,
        current
    );
}

class SafeReleaseGuard
{
public:
    explicit SafeReleaseGuard(DeviceHandler* handle)
        : handle_(handle)
    {
    }

    void arm(uint16_t return_position)
    {
        return_position_ = return_position;
        armed_ = true;
    }

    void disarm()
    {
        armed_ = false;
    }

    ~SafeReleaseGuard()
    {
        if (armed_)
        {
            send_index_current(handle_, 0);
            stark_set_finger_position_with_speed(
                handle_,
                RIGHT_HAND_ID,
                STARK_FINGER_ID_INDEX,
                return_position_,
                PREPOSITION_SPEED
            );
        }
    }

private:
    DeviceHandler* handle_;
    uint16_t return_position_{0};
    bool armed_{false};
};

class ExperimentLog
{
public:
    explicit ExperimentLog(const std::string& prefix)
        : csv_path_(prefix + ".csv"),
          meta_path_(prefix + "_meta.txt")
    {
        const std::filesystem::path parent =
            std::filesystem::path(prefix).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        if (std::filesystem::exists(csv_path_)
            || std::filesystem::exists(meta_path_))
        {
            throw std::runtime_error(
                "Output files already exist; choose a new prefix"
            );
        }

        csv_.open(csv_path_);
        meta_.open(meta_path_);
        if (!csv_ || !meta_)
            throw std::runtime_error("Failed to open output files");

        csv_
            << "wall_time,monotonic_ms,phase,"
            << "q_des_norm,q_norm,dq_norm,error_norm,"
            << "applied_current_cmd_raw,next_current_cmd_raw,"
            << "current_feedback_raw,motor_state,"
            << "read_latency_us,loop_dt_us";

        for (const char* name : MOTOR_NAMES)
            csv_ << ",position_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",speed_" << name;

        csv_ << "\n";
        start_ = std::chrono::steady_clock::now();
        previous_ = start_;
    }

    template <typename Value>
    void meta(const std::string& key, const Value& value)
    {
        meta_ << key << "=" << value << "\n";
        meta_.flush();
    }

    void sample(
        const std::string& phase,
        double q_des,
        int16_t applied_current_command,
        int16_t next_current_command,
        const Sample& data
    )
    {
        const auto now = std::chrono::steady_clock::now();
        const auto monotonic_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start_
            ).count();
        const auto loop_dt_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now - previous_
            ).count();
        previous_ = now;

        const double q = data.positions[INDEX] / 1000.0;
        const double dq = data.speeds[INDEX] / 1000.0;

        csv_
            << wall_time_iso8601() << ","
            << monotonic_ms << ","
            << phase << ","
            << std::fixed << std::setprecision(6)
            << q_des << ","
            << q << ","
            << dq << ","
            << q_des - q << ","
            << applied_current_command << ","
            << next_current_command << ","
            << data.currents[INDEX] << ","
            << static_cast<int>(data.states[INDEX]) << ","
            << data.read_latency_us << ","
            << loop_dt_us;

        for (uint16_t value : data.positions)
            csv_ << "," << value;
        for (int16_t value : data.speeds)
            csv_ << "," << value;

        csv_ << "\n";
        csv_.flush();
    }

    const std::string& csv_path() const
    {
        return csv_path_;
    }

    const std::string& meta_path() const
    {
        return meta_path_;
    }

private:
    std::string csv_path_;
    std::string meta_path_;
    std::ofstream csv_;
    std::ofstream meta_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point previous_;
};

bool validate_other_motors(
    const Sample& sample,
    const std::array<uint16_t, MOTOR_COUNT>& reference,
    std::string& reason
)
{
    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
    {
        if (unsafe_state(sample.states[i]))
        {
            reason = std::string("unsafe_state_") + MOTOR_NAMES[i];
            return false;
        }

        if (
            i != INDEX
            && sample.states[i] != MOTOR_STATE_IDLE
        )
        {
            reason = std::string("non_index_state_") + MOTOR_NAMES[i];
            return false;
        }

        if (
            i != INDEX
            && (
                std::abs(
                    static_cast<int>(sample.positions[i])
                    - static_cast<int>(reference[i])
                ) > OTHER_POSITION_DELTA_STOP
                || std::abs(static_cast<int>(sample.speeds[i]))
                    > OTHER_SPEED_STOP
            )
        )
        {
            reason = std::string("non_index_motion_") + MOTOR_NAMES[i];
            return false;
        }
    }

    return true;
}

bool read_stable_baseline(
    DeviceHandler* handle,
    ExperimentLog& log,
    Sample& last,
    std::string& reason
)
{
    Sample first;
    if (!read_sample(handle, first))
    {
        reason = "baseline_read_failure";
        return false;
    }

    const auto reference = first.positions;
    uint16_t index_min = first.positions[INDEX];
    uint16_t index_max = first.positions[INDEX];
    const auto end = std::chrono::steady_clock::now() + BASELINE_DURATION;
    auto next = std::chrono::steady_clock::now();

    while (
        !stop_requested.load()
        && std::chrono::steady_clock::now() < end
    )
    {
        Sample sample;
        if (!read_sample(handle, sample))
        {
            reason = "baseline_read_failure";
            return false;
        }

        last = sample;
        log.sample(
            "baseline",
            sample.positions[INDEX] / 1000.0,
            0,
            0,
            sample
        );

        if (!validate_other_motors(sample, reference, reason))
            return false;

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            if (
                sample.states[i] != MOTOR_STATE_IDLE
                || std::abs(static_cast<int>(sample.speeds[i]))
                    > ZERO_SPEED_TOLERANCE
            )
            {
                reason = std::string("baseline_not_idle_")
                    + MOTOR_NAMES[i];
                return false;
            }
        }

        if (sample.read_latency_us > CONTROL_DEADLINE.count() * 1000)
        {
            reason = "baseline_read_deadline";
            return false;
        }

        index_min = std::min(index_min, sample.positions[INDEX]);
        index_max = std::max(index_max, sample.positions[INDEX]);

        next += STATUS_PERIOD;
        std::this_thread::sleep_until(next);
    }

    if (stop_requested.load() || index_max - index_min > 5)
    {
        reason = stop_requested.load()
            ? "signal"
            : "baseline_position_unstable";
        return false;
    }

    return true;
}

bool move_index_position(
    DeviceHandler* handle,
    ExperimentLog& log,
    const std::string& phase,
    uint16_t target,
    const std::array<uint16_t, MOTOR_COUNT>& reference,
    std::string& reason,
    bool cleanup = false
)
{
    stark_set_finger_position_with_speed(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX,
        target,
        PREPOSITION_SPEED
    );

    const auto end = std::chrono::steady_clock::now() + POSITION_TIMEOUT;
    auto next = std::chrono::steady_clock::now();
    int stable_samples = 0;

    while (
        (cleanup || !stop_requested.load())
        && std::chrono::steady_clock::now() < end
    )
    {
        Sample sample;
        if (!read_sample(handle, sample))
        {
            reason = phase + "_read_failure";
            return false;
        }

        log.sample(phase, target / 1000.0, 0, 0, sample);

        if (!validate_other_motors(sample, reference, reason))
            return false;

        if (
            !cleanup
            && (
                sample.positions[INDEX] > MAX_INDEX_POSITION
                || std::abs(static_cast<int>(sample.speeds[INDEX]))
                    > SPEED_STOP
            )
        )
        {
            reason = phase + "_bound";
            return false;
        }

        if (
            std::abs(
                static_cast<int>(sample.positions[INDEX])
                - static_cast<int>(target)
            ) <= 5
            && std::abs(static_cast<int>(sample.speeds[INDEX]))
                <= ZERO_SPEED_TOLERANCE
            && sample.states[INDEX] == MOTOR_STATE_IDLE
        )
        {
            ++stable_samples;
            if (stable_samples >= 5)
                return true;
        }
        else
        {
            stable_samples = 0;
        }

        if (
            sample.states[INDEX] == MOTOR_STATE_STALL
            && stable_samples == 0
        )
        {
            reason = phase + "_motor_stall";
            return false;
        }

        next += STATUS_PERIOD;
        std::this_thread::sleep_until(next);
    }

    reason = (!cleanup && stop_requested.load())
        ? "signal"
        : phase + "_timeout";
    return false;
}

bool settle_zero_current(
    DeviceHandler* handle,
    ExperimentLog& log,
    const std::string& phase,
    const std::array<uint16_t, MOTOR_COUNT>& reference,
    Sample& last,
    std::string& reason,
    bool cleanup = false
)
{
    send_index_current(handle, 0);

    const auto end = std::chrono::steady_clock::now() + ZERO_TIMEOUT;
    auto next = std::chrono::steady_clock::now();
    int stable_samples = 0;

    while (
        (cleanup || !stop_requested.load())
        && std::chrono::steady_clock::now() < end
    )
    {
        Sample sample;
        if (!read_sample(handle, sample))
        {
            reason = phase + "_read_failure";
            return false;
        }

        last = sample;
        log.sample(
            phase,
            sample.positions[INDEX] / 1000.0,
            0,
            0,
            sample
        );

        if (!validate_other_motors(sample, reference, reason))
            return false;

        if (
            sample.positions[INDEX] > MAX_INDEX_POSITION
            || std::abs(static_cast<int>(sample.speeds[INDEX])) > SPEED_STOP
            || std::abs(static_cast<int>(sample.currents[INDEX]))
                > CURRENT_FEEDBACK_LIMIT
            || sample.states[INDEX] == MOTOR_STATE_STALL
        )
        {
            reason = phase + "_bound";
            return false;
        }

        if (
            std::abs(static_cast<int>(sample.currents[INDEX]))
                <= ZERO_CURRENT_TOLERANCE
            && std::abs(static_cast<int>(sample.speeds[INDEX]))
                <= ZERO_SPEED_TOLERANCE
            && sample.states[INDEX] == MOTOR_STATE_IDLE
        )
        {
            ++stable_samples;
        }
        else
        {
            stable_samples = 0;
        }

        next += STATUS_PERIOD;
        std::this_thread::sleep_until(next);
    }

    if (stable_samples < 5)
    {
        reason = (!cleanup && stop_requested.load())
            ? "signal"
            : phase + "_timeout";
        return false;
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    const auto parsed = parse_options(argc, argv);
    if (!parsed)
        return 2;
    const Options options = *parsed;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef SIGHUP
    std::signal(SIGHUP, signal_handler);
#endif

    try
    {
        init_logging(LogLevel::LOG_LEVEL_ERROR);

        ProcessLock process_lock;
        if (!process_lock.acquire())
        {
            std::cerr << "Another direct hand test holds the lock.\n";
            return 3;
        }

        ExperimentLog log(options.output_prefix);
        log.meta("controller_version", "index_impedance_v9");
        log.meta("started_at", wall_time_iso8601());
        log.meta("port", options.port);
        log.meta("slave_id", static_cast<int>(RIGHT_HAND_ID));
        log.meta("kp", KP);
        log.meta("kd", KD);
        log.meta("bilateral_current", 1);
        log.meta("current_limit_raw", CURRENT_COMMAND_LIMIT);
        log.meta("target_delta_raw", TARGET_DELTA);
        log.meta("closing_speed_soft_limit", CLOSING_SPEED_SOFT_LIMIT);
        log.meta("closing_speed_zero_current", CLOSING_SPEED_ZERO_CURRENT);
        log.meta("speed_watchdog_limit", SPEED_STOP / 1000.0);
        log.meta("h0_gate_meta", options.h0_gate_meta);

        DeviceHandler* handle =
            modbus_open(options.port.c_str(), BAUDRATE);
        if (!handle)
        {
            log.meta("result", "BLOCKED_PORT_OPEN");
            std::cerr << "Failed to open " << options.port << "\n";
            return 4;
        }

        DeviceGuard device_guard(handle);
        SafeReleaseGuard release_guard(handle);

        using InfoPtr =
            std::unique_ptr<CDeviceInfo, decltype(&free_device_info)>;
        InfoPtr info(
            read_device_info_with_retry(handle),
            &free_device_info
        );

        if (!info)
        {
            log.meta("result", "BLOCKED_DEVICE_INFO");
            return 5;
        }

        const std::string serial = safe_text(info->serial_number);
        const std::string firmware = safe_text(info->firmware_version);
        log.meta("serial_number", serial);
        log.meta("firmware_version", firmware);

        if (
            serial != options.expected_serial
            || !is_right_sku(info->sku_type)
            || !is_revo2(info->hardware_type)
        )
        {
            log.meta("result", "BLOCKED_DEVICE_MISMATCH");
            std::cerr << "RIGHT Revo2 identity check failed.\n";
            return 6;
        }

        stark_set_finger_unit_mode(
            handle,
            RIGHT_HAND_ID,
            FINGER_UNIT_MODE_NORMALIZED
        );
        const FingerUnitMode unit_mode =
            stark_get_finger_unit_mode(handle, RIGHT_HAND_ID);
        const bool turbo_enabled =
            stark_get_turbo_mode_enabled(handle, RIGHT_HAND_ID);
        const uint16_t max_current =
            stark_get_finger_max_current(
                handle,
                RIGHT_HAND_ID,
                STARK_FINGER_ID_INDEX
            );
        const uint16_t protected_current =
            stark_get_finger_protected_current(
                handle,
                RIGHT_HAND_ID,
                STARK_FINGER_ID_INDEX
            );

        log.meta("unit_mode", static_cast<int>(unit_mode));
        log.meta("turbo_enabled", turbo_enabled ? 1 : 0);
        log.meta("max_current_index", max_current);
        log.meta("protected_current_index", protected_current);

        const auto h0_meta = read_meta(options.h0_gate_meta);
        if (!h0_gate_passed(
                h0_meta,
                serial,
                firmware,
                max_current,
                protected_current))
        {
            log.meta("result", "BLOCKED_H0_GATE");
            std::cerr
                << "The successful visible-current H0 gate is missing, "
                << "stale, or did not pass.\n";
            return 7;
        }

        log.meta("h0_gate_passed", 1);
        log.meta("h0_probe_version", h0_meta.at("test"));
        log.meta("h0_result", h0_meta.at("result"));
        log.meta(
            "h0_close_current",
            h0_meta.at("close_current")
        );
        log.meta(
            "h0_visible_target",
            h0_meta.at("visible_target")
        );

        const uint16_t current_limit_ma = static_cast<uint16_t>(
            std::ceil(
                CURRENT_COMMAND_LIMIT
                * static_cast<double>(max_current)
                / 1000.0
            )
        );
        log.meta("current_limit_ma", current_limit_ma);

        if (
            unit_mode != FINGER_UNIT_MODE_NORMALIZED
            || turbo_enabled
            || max_current == 0
            || protected_current < current_limit_ma
        )
        {
            log.meta("result", "BLOCKED_CONFIGURATION");
            std::cerr << "Unit mode, Turbo, or current limits are unsafe.\n";
            return 8;
        }

        Sample baseline;
        std::string reason;
        if (!read_stable_baseline(handle, log, baseline, reason))
        {
            log.meta("result", "BLOCKED_BASELINE");
            log.meta("reason", reason);
            return 9;
        }

        const auto reference_positions = baseline.positions;
        const uint16_t start_position = baseline.positions[INDEX];

        if (
            start_position > 50
            || baseline.states[INDEX] != MOTOR_STATE_IDLE
        )
        {
            log.meta("result", "BLOCKED_INITIAL_STATE");
            log.meta("reason", reason);
            return 10;
        }

        release_guard.arm(start_position);

        std::cout
            << "Prepositioning Index to q=0.050 in position mode.\n";

        if (!move_index_position(
                handle,
                log,
                "preposition",
                PREPOSITION,
                reference_positions,
                reason))
        {
            log.meta("result", "PREPOSITION_FAILED");
            log.meta("reason", reason);
            return 11;
        }

        Sample settled;
        if (!settle_zero_current(
                handle,
                log,
                "zero_before_impedance",
                reference_positions,
                settled,
                reason))
        {
            log.meta("result", "ZERO_BEFORE_IMPEDANCE_FAILED");
            log.meta("reason", reason);
            return 12;
        }

        const uint16_t q_base_raw = settled.positions[INDEX];
        const uint16_t q_target_raw = q_base_raw + TARGET_DELTA;

        if (q_target_raw > MAX_INDEX_POSITION)
        {
            log.meta("result", "BLOCKED_TARGET_BOUND");
            return 13;
        }

        log.meta("q_base_raw", q_base_raw);
        log.meta("q_target_raw", q_target_raw);

        std::cout
            << "Impedance ramp: q_base=" << q_base_raw
            << " q_target=" << q_target_raw
            << " current_limit=" << CURRENT_COMMAND_LIMIT << "\n";

        const double q_base = q_base_raw / 1000.0;
        const double q_target = q_target_raw / 1000.0;
        double q_des = q_base;
        double filtered_dq = 0.0;
        int16_t previous_command = 0;
        int feedback_mismatch_count = 0;
        int peak_command = 0;
        int peak_feedback = 0;
        int peak_speed = 0;
        uint16_t peak_position = q_base_raw;
        bool movement_detected = false;
        bool any_actuation_detected = false;
        int print_counter = 0;
        std::optional<std::chrono::steady_clock::time_point>
            high_command_since;
        uint16_t high_command_start_position = q_base_raw;
        auto last_control_step = std::chrono::steady_clock::now();

        auto run_control_step = [&](
            const std::string& phase,
            bool ramp
        ) -> bool
        {
            const auto step_start = std::chrono::steady_clock::now();
            Sample sample;

            if (!read_sample(handle, sample))
            {
                reason = phase + "_read_failure";
                return false;
            }

            const auto after_read = std::chrono::steady_clock::now();
            if (
                sample.read_latency_us
                    > CONTROL_DEADLINE.count() * 1000
                || after_read - step_start > CONTROL_DEADLINE
            )
            {
                reason = phase + "_status_deadline";
                return false;
            }

            if (!validate_other_motors(sample, reference_positions, reason))
                return false;

            if (
                sample.positions[INDEX] + 30 < q_base_raw
                || sample.positions[INDEX] > MAX_INDEX_POSITION
                || std::abs(static_cast<int>(sample.speeds[INDEX]))
                    > SPEED_STOP
                || std::abs(static_cast<int>(sample.currents[INDEX]))
                    > CURRENT_FEEDBACK_LIMIT
                || sample.states[INDEX] == MOTOR_STATE_STALL
            )
            {
                reason = phase + "_watchdog_bound";
                return false;
            }

            if (ramp)
            {
                const double dt = std::clamp(
                    std::chrono::duration<double>(
                        step_start - last_control_step
                    ).count(),
                    0.0,
                    std::chrono::duration<double>(
                        CONTROL_DEADLINE
                    ).count()
                );
                q_des = std::min(
                    q_target,
                    q_des + RAMP_RATE * dt
                );
            }

            const double q = sample.positions[INDEX] / 1000.0;
            const double dq = sample.speeds[INDEX] / 1000.0;
            filtered_dq += EMA_ALPHA * (dq - filtered_dq);

            const double current_limit_norm =
                CURRENT_COMMAND_LIMIT / 1000.0;
            double current_norm =
                KP * (q_des - q) - KD * filtered_dq;

            if (dq > CLOSING_SPEED_SOFT_LIMIT)
            {
                const double speed_cap = current_limit_norm * std::clamp(
                    (CLOSING_SPEED_ZERO_CURRENT - dq)
                        / (
                            CLOSING_SPEED_ZERO_CURRENT
                            - CLOSING_SPEED_SOFT_LIMIT
                        ),
                    -1.0,
                    1.0
                );
                current_norm = std::min(current_norm, speed_cap);
            }

            current_norm = std::clamp(
                current_norm,
                -current_limit_norm,
                current_limit_norm
            );
            const int16_t command = static_cast<int16_t>(
                std::lround(current_norm * 1000.0)
            );

            if (
                std::abs(
                    static_cast<int>(sample.currents[INDEX])
                    - static_cast<int>(previous_command)
                ) > 35
            )
            {
                ++feedback_mismatch_count;
            }
            else
            {
                feedback_mismatch_count = 0;
            }

            if (feedback_mismatch_count >= 4)
            {
                reason = phase + "_current_feedback_mismatch";
                return false;
            }

            any_actuation_detected |=
                std::abs(
                    static_cast<int>(sample.positions[INDEX])
                    - static_cast<int>(q_base_raw)
                ) >= 5;

            if (any_actuation_detected)
                high_command_since.reset();

            if (high_command_since)
            {
                const bool actuation_seen =
                    std::abs(
                        static_cast<int>(sample.positions[INDEX])
                        - static_cast<int>(high_command_start_position)
                    ) >= 5
                    || std::abs(
                        static_cast<int>(sample.speeds[INDEX])
                    ) >= 20;

                if (actuation_seen)
                {
                    any_actuation_detected = true;
                    high_command_since.reset();
                }
                else if (
                    step_start - *high_command_since
                    >= std::chrono::milliseconds(500)
                )
                {
                    reason = phase + "_no_actuation";
                    return false;
                }
            }

            send_index_current(handle, command);
            log.sample(
                phase,
                q_des,
                previous_command,
                command,
                sample
            );

            if (++print_counter % 10 == 0)
            {
                std::cout
                    << "\r" << phase
                    << " q_des=" << std::fixed << std::setprecision(3)
                    << q_des
                    << " q=" << q
                    << " dq=" << std::showpos << dq << std::noshowpos
                    << " current_cmd=" << command
                    << " current_fb=" << sample.currents[INDEX]
                    << "   "
                    << std::flush;
            }
            previous_command = command;
            last_control_step = step_start;

            if (
                std::abs(static_cast<int>(command)) >= 90
                && !any_actuation_detected
                && !high_command_since
            )
            {
                high_command_since = step_start;
                high_command_start_position = sample.positions[INDEX];
            }
            else if (std::abs(static_cast<int>(command)) < 90)
            {
                high_command_since.reset();
            }

            peak_command = std::max(
                peak_command,
                std::abs(static_cast<int>(command))
            );
            peak_feedback = std::max(
                peak_feedback,
                std::abs(static_cast<int>(sample.currents[INDEX]))
            );
            peak_speed = std::max(
                peak_speed,
                std::abs(static_cast<int>(sample.speeds[INDEX]))
            );
            peak_position = std::max(
                peak_position,
                sample.positions[INDEX]
            );
            movement_detected |=
                sample.positions[INDEX]
                    >= q_base_raw + MIN_IMPEDANCE_TRAVEL;

            const auto elapsed = std::chrono::steady_clock::now() - step_start;
            if (elapsed > CONTROL_DEADLINE)
            {
                reason = phase + "_loop_deadline";
                return false;
            }

            return true;
        };

        const auto armed_end =
            std::chrono::steady_clock::now() + ARMED_DURATION;
        auto next = std::chrono::steady_clock::now();
        auto wait_control_period = [&]()
        {
            next += CONTROL_PERIOD;
            const auto now = std::chrono::steady_clock::now();
            if (now > next)
                next = now + CONTROL_PERIOD;
            std::this_thread::sleep_until(next);
        };

        while (
            !stop_requested.load()
            && std::chrono::steady_clock::now() < armed_end
        )
        {
            if (!run_control_step("armed", false))
                break;
            wait_control_period();
        }

        const auto ramp_end =
            std::chrono::steady_clock::now() + RAMP_DURATION;

        while (
            reason.empty()
            && !stop_requested.load()
            && std::chrono::steady_clock::now() < ramp_end
            && q_des < q_target
        )
        {
            if (!run_control_step("impedance_ramp", true))
                break;
            wait_control_period();
        }

        if (
            reason.empty()
            && !stop_requested.load()
            && q_des < q_target
        )
        {
            reason = "impedance_ramp_timeout";
        }

        if (reason.empty())
            q_des = q_target;

        const auto hold_end =
            std::chrono::steady_clock::now() + HOLD_DURATION;

        if (reason.empty())
        {
            std::cout
                << "\nImpedance hold: 5 s.\n";
        }

        while (
            reason.empty()
            && !stop_requested.load()
            && std::chrono::steady_clock::now() < hold_end
        )
        {
            if (!run_control_step("impedance_hold", false))
                break;
            wait_control_period();
        }

        if (stop_requested.load() && reason.empty())
            reason = "signal";
        if (!movement_detected && reason.empty())
            reason = "insufficient_impedance_travel";
        if (peak_command < 50 && reason.empty())
            reason = "insufficient_impedance_output";

        std::cout << "\nReturning Index in position mode.\n";
        std::string cleanup_reason;
        const bool returned = move_index_position(
                handle,
                log,
                "return_position",
                start_position,
                reference_positions,
                cleanup_reason,
                true
            );

        Sample shutdown_zero;
        std::string zero_reason;
        const bool zero_confirmed = returned && settle_zero_current(
                handle,
                log,
                "zero_after_return",
                reference_positions,
                shutdown_zero,
                zero_reason,
                true
            );

        if (!returned)
        {
            if (!reason.empty())
                reason += "; ";
            reason += "cleanup=" + cleanup_reason;
        }
        else if (!zero_confirmed)
        {
            if (!reason.empty())
                reason += "; ";
            reason += "cleanup=" + zero_reason;
        }

        if (returned && zero_confirmed)
            release_guard.disarm();

        log.meta("peak_command_raw", peak_command);
        log.meta("peak_feedback_raw", peak_feedback);
        log.meta("peak_speed_raw", peak_speed);
        log.meta("peak_position_raw", peak_position);
        log.meta("movement_detected", movement_detected ? 1 : 0);
        log.meta("returned_open", returned ? 1 : 0);
        log.meta("zero_confirmed", zero_confirmed ? 1 : 0);

        if (!reason.empty())
        {
            log.meta("result", "ABORTED");
            log.meta("reason", reason);
            std::cerr << "ABORTED: " << reason << "\n";
            std::cerr << "CSV:  " << log.csv_path() << "\n";
            std::cerr << "Meta: " << log.meta_path() << "\n";
            return 13;
        }

        log.meta("result", "SAFE_RUN_PASS");

        std::cout
            << "RESULT=SAFE_RUN_PASS\n"
            << "peak_command_raw=" << peak_command << "\n"
            << "peak_feedback_raw=" << peak_feedback << "\n"
            << "peak_speed_raw=" << peak_speed << "\n"
            << "peak_position_raw=" << peak_position << "\n"
            << "CSV:  " << log.csv_path() << "\n"
            << "Meta: " << log.meta_path() << "\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Fatal error: " << error.what() << "\n";
        return 1;
    }
}
