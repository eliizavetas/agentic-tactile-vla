#include "stark-sdk.h"

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

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

constexpr int16_t CLOSE_CURRENT = 100;
constexpr uint16_t PREPOSITION = 50;
constexpr uint16_t VISIBLE_TARGET = 220;
constexpr uint16_t POSITION_SPEED = 80;
constexpr uint16_t POSITION_TOLERANCE = 5;
constexpr uint16_t HARD_POSITION_LIMIT = 300;
constexpr int16_t HARD_SPEED_LIMIT = 700;
constexpr int STABLE_SAMPLES = 5;
constexpr int BASELINE_SAMPLES = 10;
constexpr int16_t ZERO_CURRENT_TOLERANCE = 10;
constexpr int16_t ZERO_SPEED_TOLERANCE = 30;

constexpr auto SAMPLE_PERIOD = std::chrono::milliseconds(20);
constexpr auto POSITION_TIMEOUT = std::chrono::seconds(5);
constexpr auto CURRENT_TIMEOUT = std::chrono::seconds(2);
constexpr auto VISIBLE_HOLD = std::chrono::seconds(2);

std::atomic<bool> stop_requested{false};

void signal_handler(int)
{
    stop_requested.store(true);
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

struct Options
{
    std::string port;
    std::string expected_serial;
    std::string output_prefix;
    bool confirmed{false};
};

bool parse_options(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--confirm-current-test")
        {
            options.confirmed = true;
        }
        else if (
            (arg == "--port"
             || arg == "--expected-serial"
             || arg == "--output-prefix")
            && i + 1 < argc
        )
        {
            const std::string value = argv[++i];
            if (arg == "--port")
                options.port = value;
            else if (arg == "--expected-serial")
                options.expected_serial = value;
            else
                options.output_prefix = value;
        }
        else
        {
            return false;
        }
    }

    return options.confirmed
        && !options.port.empty()
        && !options.expected_serial.empty()
        && !options.output_prefix.empty();
}

void print_usage(const char* executable)
{
    std::cout
        << "Usage:\n  " << executable
        << " --confirm-current-test"
        << " --port <device>"
        << " --expected-serial <serial>"
        << " --output-prefix <path>\n";
}

struct Sample
{
    std::array<uint16_t, MOTOR_COUNT> positions{};
    std::array<int16_t, MOTOR_COUNT> speeds{};
    std::array<int16_t, MOTOR_COUNT> currents{};
    std::array<uint8_t, MOTOR_COUNT> states{};
};

class CsvLog
{
public:
    explicit CsvLog(const std::string& prefix)
        : csv_path_(prefix + ".csv"),
          meta_path_(prefix + "_meta.txt"),
          csv_(csv_path_),
          meta_(meta_path_),
          start_(std::chrono::steady_clock::now())
    {
        if (!csv_ || !meta_)
            throw std::runtime_error("Failed to create output files");

        csv_
            << "monotonic_ms,phase,command_current,"
            << "index_position,index_speed,index_current,index_state\n";
    }

    void sample(
        const std::string& phase,
        int16_t command,
        const Sample& value
    )
    {
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::milliseconds
        >(std::chrono::steady_clock::now() - start_).count();

        csv_
            << elapsed << ','
            << phase << ','
            << command << ','
            << value.positions[INDEX] << ','
            << value.speeds[INDEX] << ','
            << value.currents[INDEX] << ','
            << static_cast<int>(value.states[INDEX]) << '\n';
        csv_.flush();
    }

    template <typename T>
    void meta(const std::string& key, const T& value)
    {
        meta_ << key << '=' << value << '\n';
        meta_.flush();
    }

    const std::string& csv_path() const { return csv_path_; }
    const std::string& meta_path() const { return meta_path_; }

private:
    std::string csv_path_;
    std::string meta_path_;
    std::ofstream csv_;
    std::ofstream meta_;
    std::chrono::steady_clock::time_point start_;
};

bool read_sample(DeviceHandler* handle, Sample& sample)
{
    CMotorStatusData* status = stark_get_motor_status(
        handle,
        RIGHT_HAND_ID
    );

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

void print_index(const std::string& phase, const Sample& sample)
{
    std::cout
        << phase
        << " q=" << sample.positions[INDEX]
        << " dq=" << sample.speeds[INDEX]
        << " current=" << sample.currents[INDEX]
        << " state=" << static_cast<int>(sample.states[INDEX])
        << '\n';
}

bool wait_position(
    DeviceHandler* handle,
    CsvLog& log,
    const std::string& phase,
    uint16_t target,
    Sample& last
)
{
    stark_set_finger_position_with_speed(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX,
        target,
        POSITION_SPEED
    );

    const auto deadline = std::chrono::steady_clock::now()
        + POSITION_TIMEOUT;
    auto next = std::chrono::steady_clock::now();
    int stable = 0;
    int samples = 0;

    while (!stop_requested.load()
           && std::chrono::steady_clock::now() < deadline)
    {
        if (!read_sample(handle, last))
            return false;

        log.sample(phase, 0, last);
        ++samples;

        if (last.positions[INDEX] > HARD_POSITION_LIMIT
            || std::abs(static_cast<int>(last.speeds[INDEX]))
                > HARD_SPEED_LIMIT)
        {
            return false;
        }

        const int error = std::abs(
            static_cast<int>(last.positions[INDEX])
            - static_cast<int>(target)
        );

        if (error <= POSITION_TOLERANCE
            && std::abs(static_cast<int>(last.speeds[INDEX])) <= 30
            && last.states[INDEX] == MOTOR_STATE_IDLE
        )
        {
            ++stable;
            if (stable >= STABLE_SAMPLES)
                return true;
        }
        else
        {
            stable = 0;
        }

        if (samples % 10 == 0)
            print_index(phase, last);

        next += SAMPLE_PERIOD;
        std::this_thread::sleep_until(next);
    }

    return false;
}

void send_zero(DeviceHandler* handle)
{
    if (handle)
    {
        stark_set_finger_current(
            handle,
            RIGHT_HAND_ID,
            STARK_FINGER_ID_INDEX,
            0
        );
    }
}

class ZeroCurrentGuard
{
public:
    explicit ZeroCurrentGuard(DeviceHandler* handle) : handle_(handle) {}

    void arm() { armed_ = true; }
    void disarm() { armed_ = false; }

    ~ZeroCurrentGuard()
    {
        if (armed_)
            send_zero(handle_);
    }

private:
    DeviceHandler* handle_;
    bool armed_{false};
};

bool collect_stable_baseline(
    DeviceHandler* handle,
    CsvLog& log,
    Sample& last
)
{
    Sample first;
    if (!read_sample(handle, first))
        return false;

    log.sample("baseline", 0, first);
    last = first;

    for (int sample_index = 1;
         sample_index < BASELINE_SAMPLES;
         ++sample_index)
    {
        std::this_thread::sleep_for(SAMPLE_PERIOD);
        if (!read_sample(handle, last))
            return false;

        log.sample("baseline", 0, last);

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            if (last.states[i] != MOTOR_STATE_IDLE
                || std::abs(static_cast<int>(last.speeds[i]))
                    > ZERO_SPEED_TOLERANCE
                || std::abs(
                    static_cast<int>(last.positions[i])
                    - static_cast<int>(first.positions[i])
                ) > 2)
            {
                return false;
            }
        }

        if (std::abs(static_cast<int>(last.currents[INDEX]))
            > ZERO_CURRENT_TOLERANCE)
        {
            return false;
        }
    }

    return true;
}

bool release_to_open(
    DeviceHandler* handle,
    CsvLog& log,
    Sample& last
)
{
    stark_set_finger_position_with_speed(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX,
        0,
        POSITION_SPEED
    );

    const auto deadline = std::chrono::steady_clock::now()
        + POSITION_TIMEOUT;
    auto next = std::chrono::steady_clock::now();
    int stable = 0;

    while (std::chrono::steady_clock::now() < deadline)
    {
        if (!read_sample(handle, last))
            return false;

        log.sample("position_open", 0, last);

        if (last.positions[INDEX] <= POSITION_TOLERANCE
            && std::abs(static_cast<int>(last.speeds[INDEX]))
                <= ZERO_SPEED_TOLERANCE
            && last.states[INDEX] == MOTOR_STATE_IDLE)
        {
            ++stable;
            if (stable >= STABLE_SAMPLES)
                return true;
        }
        else
        {
            stable = 0;
        }

        next += SAMPLE_PERIOD;
        std::this_thread::sleep_until(next);
    }

    return false;
}

bool confirm_zero_current(
    DeviceHandler* handle,
    CsvLog& log,
    Sample& last
)
{
    send_zero(handle);

    auto next = std::chrono::steady_clock::now();
    int stable = 0;
    for (int sample_index = 0; sample_index < 30; ++sample_index)
    {
        if (!read_sample(handle, last))
            return false;

        log.sample("final_zero", 0, last);

        if (std::abs(static_cast<int>(last.currents[INDEX]))
                <= ZERO_CURRENT_TOLERANCE
            && std::abs(static_cast<int>(last.speeds[INDEX]))
                <= ZERO_SPEED_TOLERANCE
            && last.states[INDEX] == MOTOR_STATE_IDLE)
        {
            ++stable;
            if (stable >= STABLE_SAMPLES)
                return true;
        }
        else
        {
            stable = 0;
        }

        next += SAMPLE_PERIOD;
        std::this_thread::sleep_until(next);
    }

    return false;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options))
    {
        print_usage(argv[0]);
        return 2;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifdef SIGHUP
    std::signal(SIGHUP, signal_handler);
#endif

    init_logging(LogLevel::LOG_LEVEL_ERROR);

    ProcessLock process_lock;
    if (!process_lock.acquire())
    {
        std::cerr << "Another H0 process holds the RIGHT hand lock\n";
        return 3;
    }

    DeviceHandler* handle = modbus_open(options.port.c_str(), BAUDRATE);
    if (!handle)
    {
        std::cerr << "Failed to open " << options.port << '\n';
        return 4;
    }
    DeviceGuard device_guard(handle);

    CsvLog log(options.output_prefix);
    log.meta("test", "h0_index_current_visible_v1");
    log.meta("port", options.port);
    log.meta("slave_id", static_cast<int>(RIGHT_HAND_ID));
    log.meta("close_current", CLOSE_CURRENT);
    log.meta("visible_target", VISIBLE_TARGET);

    CDeviceInfo* info = stark_get_device_info(handle, RIGHT_HAND_ID);
    if (!info)
    {
        std::cerr << "Failed to read RIGHT device info\n";
        return 5;
    }

    const std::string serial = info->serial_number
        ? info->serial_number
        : "";
    const std::string firmware = info->firmware_version
        ? info->firmware_version
        : "";
    const int hardware = info->hardware_type;
    const int sku = info->sku_type;
    free_device_info(info);

    log.meta("serial", serial);
    log.meta("firmware", firmware);
    log.meta("hardware", hardware);
    log.meta("sku", sku);

    if (serial != options.expected_serial || hardware != 5 || sku != 1)
    {
        std::cerr << "RIGHT hand identity check failed\n";
        return 6;
    }

    if (stark_get_turbo_mode_enabled(handle, RIGHT_HAND_ID))
    {
        std::cerr << "Turbo must be OFF\n";
        return 7;
    }

    stark_set_finger_unit_mode(
        handle,
        RIGHT_HAND_ID,
        FINGER_UNIT_MODE_NORMALIZED
    );

    if (stark_get_finger_unit_mode(handle, RIGHT_HAND_ID)
        != FINGER_UNIT_MODE_NORMALIZED)
    {
        std::cerr << "Normalized unit mode was not confirmed\n";
        return 8;
    }

    const uint16_t max_current = stark_get_finger_max_current(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX
    );
    const uint16_t protected_current = stark_get_finger_protected_current(
        handle,
        RIGHT_HAND_ID,
        STARK_FINGER_ID_INDEX
    );
    log.meta("index_max_current_mA", max_current);
    log.meta("index_protected_current_mA", protected_current);

    const uint16_t commanded_current_mA = static_cast<uint16_t>(
        (static_cast<uint32_t>(CLOSE_CURRENT) * max_current + 999)
        / 1000
    );
    log.meta("commanded_current_estimate_mA", commanded_current_mA);

    if (max_current == 0
        || protected_current == 0
        || commanded_current_mA > protected_current)
    {
        std::cerr << "Unsafe/invalid current configuration\n";
        return 9;
    }

    Sample last;
    if (!collect_stable_baseline(handle, log, last))
    {
        std::cerr << "Stable idle baseline was not confirmed\n";
        return 10;
    }

    const auto baseline_positions = last.positions;

    if (last.positions[INDEX] > 100)
    {
        std::cerr << "Index must start near open position\n";
        return 11;
    }

    bool current_target_reached = false;
    bool success = false;
    std::string reason;
    ZeroCurrentGuard zero_guard(handle);

    std::cout
        << "RIGHT Revo2 confirmed: " << serial
        << ", firmware " << firmware << '\n'
        << "Index max/protected current: "
        << max_current << '/' << protected_current << " mA\n"
        << "Command +100 corresponds to approximately "
        << commanded_current_mA << " mA in normalized mode.\n"
        << "No object. Keep both hands clear.\n"
        << "Prepositioning Index to q=0.05...\n";

    if (!wait_position(
            handle,
            log,
            "preposition",
            PREPOSITION,
            last))
    {
        reason = "preposition_failed";
    }
    else
    {
        std::cout
            << "Current close: +100 until q=0.22.\n"
            << "The finger will then hold q=0.22 for 2 seconds.\n";

        zero_guard.arm();
        stark_set_finger_current(
            handle,
            RIGHT_HAND_ID,
            STARK_FINGER_ID_INDEX,
            CLOSE_CURRENT
        );

        const auto deadline = std::chrono::steady_clock::now()
            + CURRENT_TIMEOUT;
        auto next = std::chrono::steady_clock::now();
        int samples = 0;

        while (!stop_requested.load()
               && std::chrono::steady_clock::now() < deadline)
        {
            if (!read_sample(handle, last))
            {
                reason = "current_status_read_failed";
                break;
            }

            log.sample("current_close", CLOSE_CURRENT, last);
            ++samples;

            bool other_motor_moved = false;
            for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
            {
                if (i != INDEX
                    && (
                        last.states[i] != MOTOR_STATE_IDLE
                        || std::abs(static_cast<int>(last.speeds[i])) > 100
                        || std::abs(
                            static_cast<int>(last.positions[i])
                            - static_cast<int>(baseline_positions[i])
                        ) > 30
                    ))
                {
                    other_motor_moved = true;
                }
            }

            if (other_motor_moved)
            {
                reason = "non_index_motion";
                break;
            }

            if (last.states[INDEX] == MOTOR_STATE_STALL
                || last.states[INDEX] == MOTOR_STATE_TURBO
                || last.states[INDEX] == MOTOR_STATE_UNKNOWN)
            {
                reason = "unsafe_index_state";
                break;
            }

            if (std::abs(static_cast<int>(last.currents[INDEX])) > 150)
            {
                reason = "current_feedback_bound";
                break;
            }

            if (last.positions[INDEX] > HARD_POSITION_LIMIT)
            {
                reason = "hard_position_limit";
                break;
            }

            if (std::abs(static_cast<int>(last.speeds[INDEX]))
                > HARD_SPEED_LIMIT)
            {
                reason = "hard_speed_limit";
                break;
            }

            if (samples % 5 == 0)
                print_index("current_close", last);

            if (last.positions[INDEX] >= VISIBLE_TARGET)
            {
                current_target_reached = true;
                print_index("CURRENT_TARGET_REACHED", last);
                break;
            }

            next += SAMPLE_PERIOD;
            std::this_thread::sleep_until(next);
        }

        if (!current_target_reached && reason.empty())
            reason = stop_requested.load()
                ? "signal"
                : "current_target_timeout";
    }

    // Position mode immediately replaces current mode and brakes at the
    // already reached coordinate. This follows BrainCo's release pattern.
    if (current_target_reached)
    {
        if (!wait_position(
                handle,
                log,
                "visible_hold",
                VISIBLE_TARGET,
                last))
        {
            success = false;
            reason = "visible_hold_failed";
        }
        else
        {
            success = true;
            const auto hold_end = std::chrono::steady_clock::now()
                + VISIBLE_HOLD;
            while (!stop_requested.load()
                   && std::chrono::steady_clock::now() < hold_end)
            {
                if (!read_sample(handle, last))
                {
                    success = false;
                    reason = "hold_status_read_failed";
                    break;
                }
                log.sample("visible_hold", 0, last);
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100)
                );
            }
        }
    }

    std::cout << "Opening Index in position mode...\n";
    const bool returned = release_to_open(handle, log, last);
    const bool zero_confirmed = confirm_zero_current(handle, log, last);
    if (zero_confirmed)
        zero_guard.disarm();

    if (!returned)
    {
        success = false;
        if (reason.empty())
            reason = "position_open_failed";
    }

    if (!zero_confirmed)
    {
        success = false;
        if (reason.empty())
            reason = "final_zero_not_confirmed";
    }

    log.meta("current_target_reached", current_target_reached ? 1 : 0);
    log.meta("returned_open", returned ? 1 : 0);
    log.meta("zero_confirmed", zero_confirmed ? 1 : 0);
    log.meta("final_position", last.positions[INDEX]);
    log.meta("result", success ? "PASS" : "FAIL");
    log.meta("reason", success ? "current_reached_visible_target" : reason);

    std::cout
        << "RESULT=" << (success ? "PASS" : "FAIL") << '\n'
        << "reason="
        << (success ? "current_reached_visible_target" : reason) << '\n'
        << "returned_open=" << (returned ? "YES" : "NO") << '\n'
        << "zero_confirmed=" << (zero_confirmed ? "YES" : "NO") << '\n'
        << "CSV:  " << log.csv_path() << '\n'
        << "Meta: " << log.meta_path() << '\n';

    return success ? 0 : 12;
}
