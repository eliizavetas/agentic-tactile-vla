#include <cstdlib>  // HAND-PATCH
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
// HAND-PATCH: the hand address is configurable through BRAINCO_GRASP_SLAVE
// (0x7f right, 0x7e left). Without the variable the right hand is addressed.
static const uint8_t RIGHT_HAND_ID = []() -> uint8_t {  // HAND-PATCH
    if (const char* e = std::getenv("BRAINCO_GRASP_SLAVE")) {  // HAND-PATCH
        return static_cast<uint8_t>(std::strtol(e, nullptr, 0));  // HAND-PATCH
    }  // HAND-PATCH
    return 0x7f;  // HAND-PATCH
}();  // HAND-PATCH
constexpr std::size_t MOTOR_COUNT = 6;

constexpr std::array<const char*, MOTOR_COUNT> MOTOR_NAMES = {
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
};

constexpr std::array<StarkFingerId, MOTOR_COUNT> FINGER_IDS = {
    STARK_FINGER_ID_THUMB,
    STARK_FINGER_ID_THUMB_AUX,
    STARK_FINGER_ID_INDEX,
    STARK_FINGER_ID_MIDDLE,
    STARK_FINGER_ID_RING,
    STARK_FINGER_ID_PINKY,
};

constexpr std::array<bool, MOTOR_COUNT> ACTIVE_MOTOR = {
    true, true, true, true, true, true,
};

constexpr std::array<uint16_t, MOTOR_COUNT> OPEN_POSITION = {
    0, 0, 0, 0, 0, 0,
};

constexpr std::array<uint16_t, MOTOR_COUNT> TARGET_POSITION = {
    700, 700, 750, 750, 750, 750,
};
constexpr std::array<uint16_t, MOTOR_COUNT> TARGET_POSITION_CYLINDER = {
    700, 800, 750, 750, 750, 750,
};

constexpr std::array<uint16_t, MOTOR_COUNT> POSITION_LIMIT = {
    800, 800, 850, 850, 850, 850,
};

constexpr std::array<uint16_t, MOTOR_COUNT> POSITION_SPEED = {
    150, 150, 100, 100, 100, 100,
};

constexpr int16_t HARD_CURRENT_LIMIT = 200;
constexpr int16_t SOFT_LOW_CURRENT_LIMIT = 150;
constexpr int16_t CURRENT_FEEDBACK_MARGIN = 70;
constexpr int16_t POSITION_MODE_CURRENT_BOUND = 550;
constexpr int16_t SPEED_WATCHDOG = 850;
constexpr int16_t ZERO_SPEED_TOLERANCE = 35;

constexpr double KP = 5.0;
constexpr double KD = 0.05;
constexpr double RAMP_RATE = 0.06;
constexpr double EMA_ALPHA = 0.30;
constexpr double CLOSING_SPEED_SOFT_LIMIT = 0.25;
constexpr double CLOSING_SPEED_ZERO_CURRENT = 0.55;

constexpr auto CONTROL_PERIOD = std::chrono::milliseconds(20);
constexpr auto STATUS_PERIOD = std::chrono::milliseconds(50);
constexpr auto CONTROL_DEADLINE = std::chrono::milliseconds(60);
constexpr auto BASELINE_DURATION = std::chrono::seconds(1);
constexpr auto POSITION_TIMEOUT = std::chrono::seconds(8);
constexpr auto RAMP_TIMEOUT = std::chrono::seconds(15);
constexpr auto HOLD_DURATION = std::chrono::seconds(8);

std::atomic<bool> stop_requested{false};

void signal_handler(int)
{
    stop_requested.store(true);
}

enum class Profile
{
    HARD,
    SOFT_SAME,
    SOFT_LOW,
};

struct ProfileSettings
{
    Profile profile;
    const char* name;
    int16_t current_limit;
};

std::optional<ProfileSettings> profile_from_text(const std::string& text)
{
    if (text == "nominal")
        return ProfileSettings{Profile::HARD, "nominal", HARD_CURRENT_LIMIT};
    if (text == "gentle")
        return ProfileSettings{Profile::SOFT_LOW, "gentle", SOFT_LOW_CURRENT_LIMIT};
    if (text == "hard")
        return ProfileSettings{Profile::HARD, "hard", HARD_CURRENT_LIMIT};
    if (text == "soft_same")
    {
        return ProfileSettings{
            Profile::SOFT_SAME,
            "soft_same",
            HARD_CURRENT_LIMIT,
        };
    }
    if (text == "soft_low")
    {
        return ProfileSettings{
            Profile::SOFT_LOW,
            "soft_low",
            SOFT_LOW_CURRENT_LIMIT,
        };
    }
    return std::nullopt;
}

struct Options
{
    bool confirmed{false};
    bool non_interactive{false};
    std::string port;
    std::string expected_serial;
    std::string h2_gate_meta;
    std::string output_prefix;
    std::string request_id;
    ProfileSettings profile{Profile::HARD, "hard", HARD_CURRENT_LIMIT};
};

void print_usage(const char* executable)
{
    std::cout
        << "Usage:\n  " << executable
        << " --confirm-spherical-test"
        << " --profile nominal|gentle|hard|soft_same|soft_low"
        << " --port <device>"
        << " --expected-serial <serial>"
        << " --h2-gate-meta <path>"
        << " --output-prefix <path>"
        << " [--request-id <id>]"
        << " [--non-interactive]\n";
}

std::optional<Options> parse_options(int argc, char** argv)
{
    Options options;
    bool profile_set = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return std::nullopt;
        }
        if (arg == "--confirm-spherical-test")
        {
            options.confirmed = true;
            continue;
        }
        if (arg == "--non-interactive")
        {
            options.non_interactive = true;
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
        if (arg == "--h2-gate-meta")
        {
            if (!read_value(options.h2_gate_meta))
                return std::nullopt;
            continue;
        }
        if (arg == "--output-prefix")
        {
            if (!read_value(options.output_prefix))
                return std::nullopt;
            continue;
        }
        if (arg == "--request-id")
        {
            if (!read_value(options.request_id))
                return std::nullopt;
            continue;
        }
        if (arg == "--profile")
        {
            std::string value;
            if (!read_value(value))
                return std::nullopt;
            const auto profile = profile_from_text(value);
            if (!profile)
            {
                std::cerr << "Unknown profile: " << value << "\n";
                return std::nullopt;
            }
            options.profile = *profile;
            profile_set = true;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return std::nullopt;
    }

    if (
        !options.confirmed
        || !profile_set
        || options.port.empty()
        || options.expected_serial.empty()
        || options.h2_gate_meta.empty()
        || options.output_prefix.empty()
    )
    {
        print_usage(argv[0]);
        return std::nullopt;
    }
    return options;
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

std::string safe_text(const char* value)
{
    return value ? std::string(value) : std::string();
}

std::unordered_map<std::string, std::string> read_meta(
    const std::string& path
)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Failed to open H2 gate meta file");

    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        const std::size_t separator = line.find('=');
        if (separator != std::string::npos)
            values[line.substr(0, separator)] = line.substr(separator + 1);
    }
    return values;
}

bool h2_gate_passed(
    const std::unordered_map<std::string, std::string>& meta,
    const std::string& serial,
    const std::string& firmware
)
{
    auto equals = [&](const std::string& key, const std::string& value)
    {
        const auto it = meta.find(key);
        return it != meta.end() && it->second == value;
    };

    return equals("controller_version", "index_impedance_v9")
        && equals("serial_number", serial)
        && equals("firmware_version", firmware)
        && equals("current_limit_raw", "200")
        && equals("returned_open", "1")
        && equals("zero_confirmed", "1")
        && equals("result", "SAFE_RUN_PASS");
}

bool is_right_sku(SkuType sku)
{
    // IDENTITY-PATCH: address 0x7e is the left hand, so LEFT SKUs are expected.
    // The check is not disabled; it only knows about the second hand.
    if (RIGHT_HAND_ID == 0x7e) {  // IDENTITY-PATCH
        return sku == SkuType::SKU_TYPE_MEDIUM_LEFT  // IDENTITY-PATCH
            || sku == SkuType::SKU_TYPE_SMALL_LEFT;  // IDENTITY-PATCH
    }  // IDENTITY-PATCH
    return sku == SkuType::SKU_TYPE_MEDIUM_RIGHT
        || sku == SkuType::SKU_TYPE_SMALL_RIGHT;
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
    explicit DeviceGuard(DeviceHandler* handle) : handle_(handle) {}
    ~DeviceGuard()
    {
        if (handle_)
            modbus_close(handle_);
    }

private:
    DeviceHandler* handle_;
};

void send_all_currents(
    DeviceHandler* handle,
    const std::array<int16_t, MOTOR_COUNT>& currents
)
{
    stark_set_finger_currents(
        handle,
        RIGHT_HAND_ID,
        currents.data(),
        currents.size()
    );
}

void send_zero_all_currents(DeviceHandler* handle)
{
    send_all_currents(handle, {});
}

void send_positions(
    DeviceHandler* handle,
    const std::array<uint16_t, MOTOR_COUNT>& positions
)
{
    stark_set_finger_positions_and_speeds(
        handle,
        RIGHT_HAND_ID,
        positions.data(),
        POSITION_SPEED.data(),
        positions.size()
    );
}

class SafeReleaseGuard
{
public:
    explicit SafeReleaseGuard(DeviceHandler* handle) : handle_(handle) {}

    void arm(const std::array<uint16_t, MOTOR_COUNT>& return_position)
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
            send_zero_all_currents(handle_);
            send_positions(handle_, return_position_);
        }
    }

private:
    DeviceHandler* handle_;
    std::array<uint16_t, MOTOR_COUNT> return_position_{};
    bool armed_{false};
};

class ExperimentLog
{
public:
    explicit ExperimentLog(const std::string& prefix)
        : csv_path_(prefix + ".csv"),
          meta_path_(prefix + "_meta.txt")
    {
        const auto parent = std::filesystem::path(prefix).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        if (
            std::filesystem::exists(csv_path_)
            || std::filesystem::exists(meta_path_)
        )
        {
            throw std::runtime_error(
                "Output files already exist; choose a new prefix"
            );
        }

        csv_.open(csv_path_);
        meta_.open(meta_path_);
        if (!csv_ || !meta_)
            throw std::runtime_error("Failed to open output files");

        csv_ << "wall_time,monotonic_ms,phase,read_latency_us,loop_dt_us";
        for (const char* name : MOTOR_NAMES)
            csv_ << ",q_des_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",position_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",speed_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",current_cmd_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",current_fb_" << name;
        for (const char* name : MOTOR_NAMES)
            csv_ << ",state_" << name;
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
        const std::array<double, MOTOR_COUNT>& q_des,
        const std::array<int16_t, MOTOR_COUNT>& current_command,
        const Sample& sample
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

        csv_ << wall_time_iso8601() << ','
             << monotonic_ms << ','
             << phase << ','
             << sample.read_latency_us << ','
             << loop_dt_us << std::fixed << std::setprecision(6);
        for (double value : q_des)
            csv_ << ',' << value;
        for (uint16_t value : sample.positions)
            csv_ << ',' << value;
        for (int16_t value : sample.speeds)
            csv_ << ',' << value;
        for (int16_t value : current_command)
            csv_ << ',' << value;
        for (int16_t value : sample.currents)
            csv_ << ',' << value;
        for (uint8_t value : sample.states)
            csv_ << ',' << static_cast<int>(value);
        csv_ << "\n";
        if (++samples_since_flush_ >= 50)
        {
            csv_.flush();
            samples_since_flush_ = 0;
        }
    }

    const std::string& csv_path() const { return csv_path_; }
    const std::string& meta_path() const { return meta_path_; }

private:
    std::string csv_path_;
    std::string meta_path_;
    std::ofstream csv_;
    std::ofstream meta_;
    int samples_since_flush_{0};
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point previous_;
};

std::array<double, MOTOR_COUNT> positions_as_q(const Sample& sample)
{
    std::array<double, MOTOR_COUNT> q{};
    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        q[i] = sample.positions[i] / 1000.0;
    return q;
}

bool validate_sample(
    const Sample& sample,
    int16_t current_feedback_bound,
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
        if (sample.positions[i] > POSITION_LIMIT[i])
        {
            reason = std::string("position_bound_") + MOTOR_NAMES[i];
            return false;
        }

        if (
            std::abs(static_cast<int>(sample.speeds[i]))
                > SPEED_WATCHDOG
        )
        {
            reason = std::string("speed_bound_") + MOTOR_NAMES[i];
            return false;
        }

        if (
            ACTIVE_MOTOR[i]
            && std::abs(static_cast<int>(sample.currents[i]))
                > current_feedback_bound
        )
        {
            reason = std::string("current_bound_") + MOTOR_NAMES[i];
            return false;
        }
    }
    if (sample.read_latency_us > CONTROL_DEADLINE.count() * 1000)
    {
        reason = "status_deadline";
        return false;
    }
    return true;
}

bool wait_positions(
    DeviceHandler* handle,
    ExperimentLog& log,
    const std::string& phase,
    const std::array<uint16_t, MOTOR_COUNT>& target,
    int16_t current_feedback_bound,
    Sample& last,
    std::string& reason,
    bool cleanup = false
)
{
    const auto end = std::chrono::steady_clock::now() + POSITION_TIMEOUT;
    auto next = std::chrono::steady_clock::now();
    int stable_samples = 0;
    std::array<int16_t, MOTOR_COUNT> zero_command{};

    while (
        (cleanup || !stop_requested.load())
        && std::chrono::steady_clock::now() < end
    )
    {
        send_positions(handle, target);

        Sample sample;
        if (!read_sample(handle, sample))
        {
            reason = phase + "_read_failure";
            return false;
        }
        last = sample;
        log.sample(phase, positions_as_q(sample), zero_command, sample);

        if (
            !cleanup
            && !validate_sample(sample, current_feedback_bound, reason)
        )
        {
            reason = phase + "_" + reason;
            return false;
        }

        bool stable = true;
        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            stable &= std::abs(
                static_cast<int>(sample.positions[i])
                - static_cast<int>(target[i])
            ) <= 12;
            stable &= std::abs(static_cast<int>(sample.speeds[i]))
                <= ZERO_SPEED_TOLERANCE;
            stable &= sample.states[i] == MOTOR_STATE_IDLE;
        }

        if (stable)
            ++stable_samples;
        else
            stable_samples = 0;
        if (stable_samples >= 4)
            return true;

        next += STATUS_PERIOD;
        std::this_thread::sleep_until(next);
    }

    reason = (!cleanup && stop_requested.load())
        ? "signal"
        : phase + "_timeout";
    return false;
}

bool read_stable_baseline(
    DeviceHandler* handle,
    ExperimentLog& log,
    int16_t current_feedback_bound,
    Sample& last,
    std::string& reason
)
{
    const auto end = std::chrono::steady_clock::now() + BASELINE_DURATION;
    auto next = std::chrono::steady_clock::now();
    std::array<int16_t, MOTOR_COUNT> zero_command{};
    std::array<uint16_t, MOTOR_COUNT> min_position{};
    std::array<uint16_t, MOTOR_COUNT> max_position{};
    bool first = true;

    while (!stop_requested.load() && std::chrono::steady_clock::now() < end)
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
            positions_as_q(sample),
            zero_command,
            sample
        );
        if (!validate_sample(sample, current_feedback_bound, reason))
        {
            reason = "baseline_" + reason;
            return false;
        }

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            if (first)
            {
                min_position[i] = sample.positions[i];
                max_position[i] = sample.positions[i];
            }
            else
            {
                min_position[i] = std::min(min_position[i], sample.positions[i]);
                max_position[i] = std::max(max_position[i], sample.positions[i]);
            }
            if (std::abs(static_cast<int>(sample.speeds[i])) > ZERO_SPEED_TOLERANCE)
            {
                reason = std::string("baseline_motion_") + MOTOR_NAMES[i];
                return false;
            }
            if (sample.states[i] != MOTOR_STATE_IDLE)
            {
                reason = std::string("baseline_not_idle_") + MOTOR_NAMES[i];
                return false;
            }
        }
        first = false;
        next += STATUS_PERIOD;
        std::this_thread::sleep_until(next);
    }

    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
    {
        if (max_position[i] - min_position[i] > 8)
        {
            reason = std::string("baseline_unstable_") + MOTOR_NAMES[i];
            return false;
        }
    }
    if (stop_requested.load())
    {
        reason = "signal";
        return false;
    }
    return true;
}

struct Metrics
{
    std::array<int, MOTOR_COUNT> peak_command{};
    std::array<int, MOTOR_COUNT> peak_feedback{};
    std::array<int, MOTOR_COUNT> peak_speed{};
    std::array<uint16_t, MOTOR_COUNT> peak_position{};
    std::array<bool, MOTOR_COUNT> moved{};
    std::array<bool, MOTOR_COUNT> contact_like{};
    std::array<int, MOTOR_COUNT> contact_counter{};
};

int count_active(const std::array<bool, MOTOR_COUNT>& values)
{
    int count = 0;
    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        count += ACTIVE_MOTOR[i] && values[i] ? 1 : 0;
    return count;
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
        log.meta("controller_version", "spherical_impedance_v3");
        log.meta("started_at", wall_time_iso8601());
        log.meta("request_id", options.request_id);
        log.meta("backend_name", "brainco_revo2_direct");
        log.meta("control_profile", "REVO_CURRENT_PROFILE");
        log.meta("grasp_shape", "SPHERICAL");
        log.meta("profile", options.profile.name);
        log.meta(
            "current_preset",
            options.profile.current_limit == SOFT_LOW_CURRENT_LIMIT
                ? "GENTLE_150"
                : "NOMINAL_200"
        );
        log.meta("port", options.port);
        log.meta("slave_id", static_cast<int>(RIGHT_HAND_ID));
        log.meta("h2_gate_meta", options.h2_gate_meta);
        log.meta("kp", KP);
        log.meta("kd", KD);
        log.meta("ramp_rate", RAMP_RATE);
        log.meta("current_limit_raw", options.profile.current_limit);
        log.meta("feedback_check", "positive_closing_current_only");
        log.meta("hold_seconds", HOLD_DURATION.count());
        log.meta("retention_observed", 0);
        log.meta("retention_status", "UNKNOWN");
        log.meta("task_success", "UNKNOWN");
        log.meta("external_retention_required", 1);
        log.meta("contact_loss_observability", "LIMITED");
        log.meta("object_identity_available", 0);

        DeviceHandler* handle = modbus_open(options.port.c_str(), BAUDRATE);
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

        const auto h2_meta = read_meta(options.h2_gate_meta);
        if (!h2_gate_passed(h2_meta, serial, firmware))
        {
            log.meta("result", "BLOCKED_H2_GATE");
            std::cerr << "A successful Index H2 v9 gate is required.\n";
            return 7;
        }
        log.meta("h2_gate_passed", 1);

        stark_set_finger_unit_mode(
            handle,
            RIGHT_HAND_ID,
            FINGER_UNIT_MODE_NORMALIZED
        );
        const FingerUnitMode unit_mode =
            stark_get_finger_unit_mode(handle, RIGHT_HAND_ID);
        const bool turbo_enabled =
            stark_get_turbo_mode_enabled(handle, RIGHT_HAND_ID);
        const uint16_t aux_lock_current =
            stark_get_thumb_aux_lock_current(handle, RIGHT_HAND_ID);
        log.meta("unit_mode", static_cast<int>(unit_mode));
        log.meta("turbo_enabled", turbo_enabled ? 1 : 0);
        log.meta("thumb_aux_lock_current_mA", aux_lock_current);

        if (
            unit_mode != FINGER_UNIT_MODE_NORMALIZED
            || turbo_enabled
            || aux_lock_current < 100
            || aux_lock_current > 500
        )
        {
            log.meta("result", "BLOCKED_CONFIGURATION");
            std::cerr << "Unit mode, Turbo, or Thumb AUX lock is unsafe.\n";
            return 8;
        }

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            const uint16_t max_current = stark_get_finger_max_current(
                handle,
                RIGHT_HAND_ID,
                FINGER_IDS[i]
            );
            const uint16_t protected_current =
                stark_get_finger_protected_current(
                    handle,
                    RIGHT_HAND_ID,
                    FINGER_IDS[i]
                );
            log.meta(std::string("max_current_mA_") + MOTOR_NAMES[i], max_current);
            log.meta(
                std::string("protected_current_mA_") + MOTOR_NAMES[i],
                protected_current
            );

            const uint16_t requested_current_mA =
                max_current == 0
                ? 0
                : static_cast<uint16_t>(std::ceil(
                    options.profile.current_limit
                    * static_cast<double>(max_current)
                    / 1000.0
                ));
            log.meta(
                std::string("requested_current_mA_") + MOTOR_NAMES[i],
                requested_current_mA
            );

            if (
                ACTIVE_MOTOR[i]
                && (
                    max_current == 0
                    || protected_current < requested_current_mA
                )
            )
            {
                log.meta("result", "BLOCKED_CURRENT_LIMITS");
                std::cerr << "Unsafe current settings for "
                          << MOTOR_NAMES[i] << "\n";
                return 9;
            }
        }

        std::string reason;
        const auto return_position = OPEN_POSITION;
        release_guard.arm(return_position);

        std::cout
            << "Resetting all six drives to open position mode. "
            << "Keep the hand empty.\n";
        Sample reset_sample;
        if (!wait_positions(
                handle,
                log,
                "reset_open_position_mode",
                OPEN_POSITION,
                POSITION_MODE_CURRENT_BOUND,
                reset_sample,
                reason))
        {
            log.meta("result", "RESET_POSITION_MODE_FAILED");
            log.meta("reason", reason);
            std::cerr << "RESET_POSITION_MODE_FAILED: " << reason << "\n";
            return 10;
        }

        Sample baseline;
        if (!read_stable_baseline(
                handle,
                log,
                POSITION_MODE_CURRENT_BOUND,
                baseline,
                reason))
        {
            log.meta("result", "BLOCKED_BASELINE");
            log.meta("reason", reason);
            std::cerr << "BLOCKED_BASELINE: " << reason << "\n";
            return 11;
        }

        std::cout
            << "\nPROFILE=" << options.profile.name
            << " current_limit=" << options.profile.current_limit << "\n"
            << "All six drives are open; there is no thumb pre-position.\n"
            << "Place the sphere in the open palm.\n"
            << "Remove hands from the grasp area.\n"
            << "Keep a soft support below: release is automatic after 8 s.\n"
            << (
                options.non_interactive
                    ? "Non-interactive start was explicitly confirmed.\n"
                    : "Press ENTER to start, or Ctrl+C to cancel.\n"
            );

        std::string confirmation;
        if (
            (!options.non_interactive
                && !std::getline(std::cin, confirmation))
            || stop_requested.load()
        )
        {
            log.meta("result", "CANCELLED_BEFORE_GRASP");
            return 12;
        }

        Sample armed_sample;
        if (!read_sample(handle, armed_sample))
        {
            log.meta("result", "ARMED_READ_FAILED");
            std::cerr << "ARMED_READ_FAILED\n";
            return 13;
        }
        if (!validate_sample(
                armed_sample,
                POSITION_MODE_CURRENT_BOUND,
                reason))
        {
            log.meta("result", "ARMED_STATE_INVALID");
            log.meta("reason", reason);
            std::cerr << "ARMED_STATE_INVALID: " << reason << "\n";
            return 13;
        }
        std::array<int16_t, MOTOR_COUNT> zero_command{};
        log.sample(
            "armed_position_mode",
            positions_as_q(armed_sample),
            zero_command,
            armed_sample
        );

        std::array<double, MOTOR_COUNT> q_des = positions_as_q(armed_sample);
        std::array<double, MOTOR_COUNT> filtered_dq{};
        std::array<int16_t, MOTOR_COUNT> previous_command{};
        std::array<int, MOTOR_COUNT> feedback_mismatch_count{};
        Metrics metrics;
        metrics.peak_position = armed_sample.positions;
        const auto q_base_raw = armed_sample.positions;
        auto last_control_step = std::chrono::steady_clock::now();
        int print_counter = 0;

        auto run_control_step = [&](const std::string& phase, bool ramp)
        {
            const auto step_start = std::chrono::steady_clock::now();
            Sample sample;
            if (!read_sample(handle, sample))
            {
                reason = phase + "_read_failure";
                return false;
            }
            if (!validate_sample(
                    sample,
                    options.profile.current_limit + CURRENT_FEEDBACK_MARGIN,
                    reason))
            {
                reason = phase + "_" + reason;
                return false;
            }

            const double dt = std::clamp(
                std::chrono::duration<double>(
                    step_start - last_control_step
                ).count(),
                0.0,
                std::chrono::duration<double>(CONTROL_DEADLINE).count()
            );

            if (ramp)
            {
                for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                {
                    if (!ACTIVE_MOTOR[i])
                        continue;
                    q_des[i] = std::min(
                        TARGET_POSITION[i] / 1000.0,
                        q_des[i] + RAMP_RATE * dt
                    );
                }
            }
            std::array<int16_t, MOTOR_COUNT> command{};
            const double current_limit_norm =
                options.profile.current_limit / 1000.0;

            for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
            {
                if (!ACTIVE_MOTOR[i])
                    continue;

                const double q = sample.positions[i] / 1000.0;
                const double dq = sample.speeds[i] / 1000.0;
                filtered_dq[i] += EMA_ALPHA * (dq - filtered_dq[i]);
                double current_norm =
                    KP * (q_des[i] - q) - KD * filtered_dq[i];

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
                command[i] = static_cast<int16_t>(
                    std::lround(current_norm * 1000.0)
                );

                const bool closing_feedback_mismatch =
                    previous_command[i] >= 60
                    && std::abs(
                        static_cast<int>(sample.currents[i])
                        - static_cast<int>(previous_command[i])
                    ) > 55;
                if (closing_feedback_mismatch)
                {
                    ++feedback_mismatch_count[i];
                }
                else
                {
                    feedback_mismatch_count[i] = 0;
                }
                if (feedback_mismatch_count[i] >= 5)
                {
                    reason = phase + "_current_feedback_mismatch_"
                        + MOTOR_NAMES[i];
                    return false;
                }

                metrics.moved[i] |= std::abs(
                    static_cast<int>(sample.positions[i])
                    - static_cast<int>(q_base_raw[i])
                ) >= 10;

                const bool contact_now =
                    q_des[i] - q >= 0.03
                    && std::abs(dq) <= 0.035
                    && command[i]
                        >= static_cast<int16_t>(
                            options.profile.current_limit * 0.80
                        );
                if (contact_now)
                    ++metrics.contact_counter[i];
                else
                    metrics.contact_counter[i] = 0;
                metrics.contact_like[i] |= metrics.contact_counter[i] >= 5;

                metrics.peak_command[i] = std::max(
                    metrics.peak_command[i],
                    std::abs(static_cast<int>(command[i]))
                );
                metrics.peak_feedback[i] = std::max(
                    metrics.peak_feedback[i],
                    std::abs(static_cast<int>(sample.currents[i]))
                );
                metrics.peak_speed[i] = std::max(
                    metrics.peak_speed[i],
                    std::abs(static_cast<int>(sample.speeds[i]))
                );
                metrics.peak_position[i] = std::max(
                    metrics.peak_position[i],
                    sample.positions[i]
                );
            }

            send_all_currents(handle, command);
            log.sample(phase, q_des, command, sample);
            previous_command = command;
            last_control_step = step_start;

            if (++print_counter % 10 == 0)
            {
                std::cout << '\r' << phase;
                for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                {
                    if (!ACTIVE_MOTOR[i])
                        continue;
                    std::cout
                        << ' ' << MOTOR_NAMES[i]
                        << "=" << sample.positions[i]
                        << "/" << command[i];
                }
                std::cout << " contacts="
                          << count_active(metrics.contact_like)
                          << "   " << std::flush;
            }

            if (std::chrono::steady_clock::now() - step_start > CONTROL_DEADLINE)
            {
                reason = phase + "_loop_deadline";
                return false;
            }
            return true;
        };

        std::cout << "Starting six-drive spherical impedance grasp.\n";
        auto next = std::chrono::steady_clock::now();
        auto wait_control_period = [&]()
        {
            next += CONTROL_PERIOD;
            const auto now = std::chrono::steady_clock::now();
            if (now > next)
                next = now + CONTROL_PERIOD;
            std::this_thread::sleep_until(next);
        };

        const auto ramp_end = std::chrono::steady_clock::now() + RAMP_TIMEOUT;
        auto targets_reached = [&]()
        {
            for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
            {
                if (
                    ACTIVE_MOTOR[i]
                    && q_des[i] < TARGET_POSITION[i] / 1000.0
                )
                    return false;
            }
            return true;
        };

        while (
            reason.empty()
            && !stop_requested.load()
            && std::chrono::steady_clock::now() < ramp_end
            && !targets_reached()
        )
        {
            if (!run_control_step("impedance_ramp", true))
                break;
            wait_control_period();
        }
        if (reason.empty() && !stop_requested.load() && !targets_reached())
            reason = "impedance_ramp_timeout";

        const auto hold_end = std::chrono::steady_clock::now() + HOLD_DURATION;
        if (reason.empty())
            std::cout << "\nHolding for 8 s. Do not touch the object.\n";

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
        const int moved_count = count_active(metrics.moved);
        const int contact_count = count_active(metrics.contact_like);
        if (reason.empty() && moved_count < 4)
            reason = "fewer_than_four_drives_moved";

        std::cout
            << "\nReleasing. Keep hands clear; the sphere may fall.\n";
        send_zero_all_currents(handle);

        Sample returned_sample;
        std::string cleanup_reason;
        const bool returned = wait_positions(
            handle,
            log,
            "return_position",
            return_position,
            options.profile.current_limit,
            returned_sample,
            cleanup_reason,
            true
        );

        if (!returned)
        {
            if (!reason.empty())
                reason += ";";
            reason += "cleanup=" + cleanup_reason;
        }
        if (returned)
            release_guard.disarm();

        log.meta("moved_drive_count", moved_count);
        log.meta("contact_like_count", contact_count);
        log.meta("returned_open", returned ? 1 : 0);
        log.meta("zero_command_sent", 1);
        log.meta("position_mode_restored", returned ? 1 : 0);
        log.meta(
            "controller_outcome",
            contact_count >= 3
                ? "CONTACT_HOLD_CANDIDATE"
                : "INSUFFICIENT_CONTACT_EVIDENCE"
        );

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            if (!ACTIVE_MOTOR[i])
                continue;
            log.meta(
                std::string("peak_command_raw_") + MOTOR_NAMES[i],
                metrics.peak_command[i]
            );
            log.meta(
                std::string("peak_feedback_raw_") + MOTOR_NAMES[i],
                metrics.peak_feedback[i]
            );
            log.meta(
                std::string("peak_speed_raw_") + MOTOR_NAMES[i],
                metrics.peak_speed[i]
            );
            log.meta(
                std::string("peak_position_raw_") + MOTOR_NAMES[i],
                metrics.peak_position[i]
            );
            log.meta(
                std::string("contact_like_") + MOTOR_NAMES[i],
                metrics.contact_like[i] ? 1 : 0
            );
        }

        if (!reason.empty())
        {
            log.meta("result", "ABORTED");
            log.meta("safety_status", "ABORTED");
            log.meta("reason", reason);
            std::cerr << "ABORTED: " << reason << "\n";
            std::cerr << "CSV:  " << log.csv_path() << "\n";
            std::cerr << "Meta: " << log.meta_path() << "\n";
            return 15;
        }

        log.meta("result", "SAFE_RUN_PASS");
        log.meta("safety_status", "OK");
        std::cout
            << "RESULT=SAFE_RUN_PASS\n"
            << "profile=" << options.profile.name << "\n"
            << "current_limit_raw=" << options.profile.current_limit << "\n"
            << "moved_drive_count=" << moved_count << "\n"
            << "contact_like_count=" << contact_count << "\n"
            << "retention_observed=NO\n"
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
