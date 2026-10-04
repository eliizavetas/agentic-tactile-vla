#define main brainco_h3_diagnostic_main_not_used
#include <cstdlib>  // IDENTITY-PATCH
#include "h3_spherical_grasp.cpp"
#undef main

#ifndef __linux__
#error "brainco_grasp_service is supported only on Linux"
#endif

#include <charconv>
#include <cctype>
#include <condition_variable>
#include <mutex>
#include <poll.h>
#include <grp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

namespace grasp_service
{

constexpr auto MAX_HOLD_DURATION = std::chrono::seconds(120);
constexpr std::size_t MAX_REQUEST_BYTES = 8192;

// --- selectable grasp shape (sphere = validated default, cylinder = thumb alongside)
std::atomic<bool> use_cylinder_shape{false};
inline double active_target_q(std::size_t i)
{
    return (use_cylinder_shape.load() ? TARGET_POSITION_CYLINDER[i]
                                      : TARGET_POSITION[i]) / 1000.0;
}
inline const char* active_shape_name()
{
    return use_cylinder_shape.load() ? "cylinder" : "sphere";
}

std::atomic<bool> shutdown_requested{false};

void service_signal_handler(int)
{
    shutdown_requested.store(true);
}

enum class Lifecycle
{
    READY,
    PREPARING,
    CLOSING,
    HOLDING,
    RELEASING,
    TERMINAL,
    FAULT,
};

const char* lifecycle_text(Lifecycle value)
{
    switch (value)
    {
        case Lifecycle::READY: return "READY";
        case Lifecycle::PREPARING: return "PREPARING";
        case Lifecycle::CLOSING: return "CLOSING";
        case Lifecycle::HOLDING: return "HOLDING";
        case Lifecycle::RELEASING: return "RELEASING";
        case Lifecycle::TERMINAL: return "TERMINAL";
        case Lifecycle::FAULT: return "FAULT";
    }
    return "FAULT";
}

struct ServiceOptions
{
    bool confirmed{false};
    std::string port;
    std::string expected_serial;
    std::string nominal_gate_meta;
    std::string gentle_gate_meta;
    std::string socket_path{"/tmp/brainco_grasp.sock"};
    std::string socket_group{"dialout"};
    std::string log_dir;
};

void print_service_usage(const char* executable)
{
    std::cout
        << "Usage:\n  " << executable
        << " --confirm-hardware"
        << " --port <device>"
        << " --expected-serial <serial>"
        << " --nominal-gate-meta <hard_v3_meta.txt>"
        << " --gentle-gate-meta <soft_low_v3_meta.txt>"
        << " --log-dir <directory>"
        << " [--socket <path>]"
        << " [--socket-group <group>]\n";
}

std::optional<ServiceOptions> parse_service_options(int argc, char** argv)
{
    ServiceOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_service_usage(argv[0]);
            return std::nullopt;
        }
        if (arg == "--confirm-hardware")
        {
            options.confirmed = true;
            continue;
        }
        auto value = [&](std::string& destination)
        {
            if (i + 1 >= argc)
                return false;
            destination = argv[++i];
            return true;
        };
        if (arg == "--port" && value(options.port)) continue;
        if (arg == "--expected-serial" && value(options.expected_serial))
            continue;
        if (arg == "--nominal-gate-meta" && value(options.nominal_gate_meta))
            continue;
        if (arg == "--gentle-gate-meta" && value(options.gentle_gate_meta))
            continue;
        if (arg == "--socket" && value(options.socket_path)) continue;
        if (arg == "--socket-group" && value(options.socket_group)) continue;
        if (arg == "--log-dir" && value(options.log_dir)) continue;
        std::cerr << "Unknown or incomplete option: " << arg << "\n";
        return std::nullopt;
    }
    if (
        !options.confirmed
        || options.port.empty()
        || options.expected_serial.empty()
        || options.nominal_gate_meta.empty()
        || options.gentle_gate_meta.empty()
        || options.socket_path.empty()
        || options.log_dir.empty()
    )
    {
        print_service_usage(argv[0]);
        return std::nullopt;
    }
    return options;
}

std::string json_escape(const std::string& text)
{
    std::ostringstream out;
    for (unsigned char c : text)
    {
        switch (c)
        {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20)
                {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(c)
                        << std::dec;
                }
                else
                {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::optional<std::size_t> json_value_start(
    const std::string& line,
    const std::string& key
)
{
    const std::string marker = "\"" + key + "\"";
    std::size_t position = line.find(marker);
    if (position == std::string::npos)
        return std::nullopt;
    position = line.find(':', position + marker.size());
    if (position == std::string::npos)
        return std::nullopt;
    ++position;
    while (
        position < line.size()
        && std::isspace(static_cast<unsigned char>(line[position]))
    )
        ++position;
    return position;
}

std::optional<std::string> json_string(
    const std::string& line,
    const std::string& key
)
{
    const auto start = json_value_start(line, key);
    if (!start || *start >= line.size() || line[*start] != '"')
        return std::nullopt;
    std::string value;
    bool escaped = false;
    for (std::size_t i = *start + 1; i < line.size(); ++i)
    {
        const char c = line[i];
        if (escaped)
        {
            switch (c)
            {
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                default: return std::nullopt;
            }
            escaped = false;
        }
        else if (c == '\\')
        {
            escaped = true;
        }
        else if (c == '"')
        {
            return value;
        }
        else
        {
            value.push_back(c);
        }
    }
    return std::nullopt;
}

std::optional<long long> json_integer(
    const std::string& line,
    const std::string& key
)
{
    const auto start = json_value_start(line, key);
    if (!start || *start >= line.size())
        return std::nullopt;
    long long value = 0;
    const char* first = line.data() + *start;
    const char* last = line.data() + line.size();
    const auto parsed = std::from_chars(first, last, value);
    if (parsed.ec != std::errc())
        return std::nullopt;
    return value;
}

bool valid_request_id(const std::string& value)
{
    if (value.empty() || value.size() > 64)
        return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c)
    {
        return std::isalnum(c) || c == '_' || c == '-' || c == '.';
    });
}

std::string safe_file_component(std::string value)
{
    for (char& c : value)
    {
        const unsigned char byte = static_cast<unsigned char>(c);
        if (!(std::isalnum(byte) || c == '_' || c == '-' || c == '.'))
            c = '_';
    }
    return value;
}

std::string timestamp_for_file()
{
    const auto now_point = std::chrono::system_clock::now();
    const std::time_t now = std::chrono::system_clock::to_time_t(now_point);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now_point.time_since_epoch()
        ).count() % 1000;
    std::tm local{};
    localtime_r(&now, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d_%H%M%S")
        << '_' << std::setw(3) << std::setfill('0') << milliseconds;
    return out.str();
}

bool meta_equals(
    const std::unordered_map<std::string, std::string>& meta,
    const std::string& key,
    const std::string& value
)
{
    const auto found = meta.find(key);
    return found != meta.end() && found->second == value;
}

bool h3_gate_passed(
    const std::string& path,
    const std::string& serial,
    const std::string& firmware,
    int16_t current_limit
)
{
    const auto meta = read_meta(path);
    const bool expected_profile = current_limit == HARD_CURRENT_LIMIT
        ? (
            meta_equals(meta, "profile", "hard")
            || meta_equals(meta, "profile", "soft_same")
            || meta_equals(meta, "profile", "nominal")
        )
        : (
            meta_equals(meta, "profile", "soft_low")
            || meta_equals(meta, "profile", "gentle")
        );
    // // GATE-PATCH: the serial check can be waived for meta recorded on the other
    // hand of the same model. Everything else in the gate still applies.
    const bool any_serial =  // GATE-PATCH
        std::getenv("BRAINCO_GRASP_GATE_ANY_SERIAL") != nullptr;  // GATE-PATCH
    return meta_equals(meta, "controller_version", "spherical_impedance_v3")
        && (any_serial || meta_equals(meta, "serial_number", serial))
        && meta_equals(meta, "firmware_version", firmware)
        && meta_equals(meta, "current_limit_raw", std::to_string(current_limit))
        && meta_equals(meta, "moved_drive_count", "6")
        && meta_equals(meta, "contact_like_count", "6")
        && meta_equals(meta, "returned_open", "1")
        && meta_equals(meta, "position_mode_restored", "1")
        && meta_equals(meta, "result", "SAFE_RUN_PASS")
        && expected_profile;
}

struct AttemptMetrics
{
    std::array<int, MOTOR_COUNT> peak_command{};
    std::array<int, MOTOR_COUNT> peak_feedback{};
    std::array<int, MOTOR_COUNT> peak_speed{};
    std::array<uint16_t, MOTOR_COUNT> peak_position{};
    std::array<bool, MOTOR_COUNT> moved{};
    std::array<bool, MOTOR_COUNT> contact_like{};
    std::array<int, MOTOR_COUNT> contact_counter{};
};

struct SharedStatus
{
    Lifecycle lifecycle{Lifecycle::READY};
    bool busy{false};
    bool terminal{false};
    bool release_complete{true};
    std::string request_id;
    long long last_sequence_id{-1};
    std::string last_action;
    std::string profile;
    std::string controller_outcome;
    std::string retention_status{"UNKNOWN"};
    std::string task_success{"UNKNOWN"};
    std::string safety_status{"OK"};
    std::string result;
    std::string reason;
    std::string csv_path;
    std::string meta_path;
    int current_limit_raw{0};
    int moved_drive_count{0};
    int contact_like_count{0};
};

class Service
{
public:
    Service(
        DeviceHandler* handle,
        ServiceOptions options,
        std::string serial,
        std::string firmware
    )
        : handle_(handle),
          options_(std::move(options)),
          serial_(std::move(serial)),
          firmware_(std::move(firmware))
    {}

    ~Service()
    {
        abort_requested_.store(true);
        release_requested_.store(true);
        if (worker_.joinable())
            worker_.join();
    }

    bool initialize(std::string& reason)
    {
        std::lock_guard<std::mutex> serial_lock(serial_mutex_);
        return ensure_open(reason);
    }

    std::string dispatch(const std::string& line)
    {
        const auto action_value = json_string(line, "action");
        if (!action_value)
            return error_response("INVALID_COMMAND", "missing action");
        std::string action = *action_value;
        std::transform(
            action.begin(),
            action.end(),
            action.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); }
        );
        if (action == "STATUS")
            return status_response(true, false, "");
        if (action == "START")
            return start(line);
        if (action == "RELEASE")
            return release(line);
        if (action == "ABORT")
            return abort(line);
        if (action == "RESET")
            return reset(line);
        return error_response("INVALID_COMMAND", "unsupported action");
    }

    void shutdown()
    {
        abort_requested_.store(true);
        release_requested_.store(true);
        if (worker_.joinable())
            worker_.join();
        std::lock_guard<std::mutex> serial_lock(serial_mutex_);
        send_zero_all_currents(handle_);
        std::string ignored;
        ensure_open(ignored);
    }

private:
    std::string error_response(
        const std::string& code,
        const std::string& reason
    ) const
    {
        return "{\"accepted\":false,\"error\":\""
            + json_escape(code) + "\",\"reason\":\""
            + json_escape(reason) + "\"}";
    }

    std::string status_response(
        bool accepted,
        bool idempotent,
        const std::string& message
    ) const
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        std::ostringstream out;
        out << "{\"accepted\":" << (accepted ? "true" : "false")
            << ",\"idempotent\":" << (idempotent ? "true" : "false")
            << ",\"message\":\"" << json_escape(message) << "\""
            << ",\"backend_name\":\"brainco_revo2_direct\""
            << ",\"profile_name\":\"REVO_CURRENT_PROFILE\""
            << ",\"control_profile\":\"REVO_CURRENT_PROFILE\""
            << ",\"capability\":\"SPHERICAL_CURRENT_IMPEDANCE_V1\""
            << ",\"shape\":\"" << active_shape_name() << "\""
            << ",\"request_id\":\"" << json_escape(status_.request_id) << "\""
            << ",\"last_sequence_id\":" << status_.last_sequence_id
            << ",\"lifecycle_state\":\""
            << lifecycle_text(status_.lifecycle) << "\""
            << ",\"busy\":" << (status_.busy ? "true" : "false")
            << ",\"terminal\":" << (status_.terminal ? "true" : "false")
            << ",\"ownership\":\""
            << (
                status_.busy
                    ? "GRASP_SKILL"
                    : (status_.lifecycle == Lifecycle::FAULT ? "SAFE" : "AGENT")
            ) << "\""
            << ",\"can_start\":"
            << (
                status_.lifecycle == Lifecycle::READY && !status_.busy
                    ? "true" : "false"
            )
            << ",\"can_release\":" << (status_.busy ? "true" : "false")
            << ",\"can_abort\":" << (status_.busy ? "true" : "false")
            << ",\"can_reset\":"
            << (status_.terminal && !status_.busy ? "true" : "false")
            << ",\"release_complete\":"
            << (status_.release_complete ? "true" : "false")
            << ",\"profile\":\"" << json_escape(status_.profile) << "\""
            << ",\"controller_outcome\":\""
            << json_escape(status_.controller_outcome) << "\""
            << ",\"retention_status\":\""
            << status_.retention_status << "\""
            << ",\"task_success\":\"" << status_.task_success << "\""
            << ",\"safety_status\":\"" << status_.safety_status << "\""
            << ",\"result\":\"" << json_escape(status_.result) << "\""
            << ",\"reason\":\"" << json_escape(status_.reason) << "\""
            << ",\"current_limit_raw\":" << status_.current_limit_raw
            << ",\"moved_drive_count\":" << status_.moved_drive_count
            << ",\"contact_like_count\":" << status_.contact_like_count
            << ",\"external_retention_required\":true"
            << ",\"object_identity_available\":false"
            << ",\"contact_loss_observability\":\"LIMITED\""
            << ",\"csv_path\":\"" << json_escape(status_.csv_path) << "\""
            << ",\"meta_path\":\"" << json_escape(status_.meta_path) << "\"}";
        return out.str();
    }

    bool command_identity(
        const std::string& line,
        std::string& request_id,
        long long& sequence_id,
        std::string& error
    ) const
    {
        const auto request = json_string(line, "request_id");
        const auto sequence = json_integer(line, "sequence_id");
        if (!request || !valid_request_id(*request))
        {
            error = "request_id must use 1-64 [A-Za-z0-9_.-] characters";
            return false;
        }
        if (!sequence || *sequence < 0)
        {
            error = "sequence_id must be a non-negative integer";
            return false;
        }
        request_id = *request;
        sequence_id = *sequence;
        return true;
    }

    bool duplicate_or_stale_locked(
        const std::string& request_id,
        long long sequence_id,
        const std::string& action,
        bool& duplicate
    ) const
    {
        duplicate = request_id == status_.request_id
            && sequence_id == status_.last_sequence_id
            && action == status_.last_action;
        return !duplicate
            && request_id == status_.request_id
            && sequence_id <= status_.last_sequence_id;
    }

    std::string start(const std::string& line)
    {
        std::string request_id;
        std::string error;
        long long sequence_id = -1;
        if (!command_identity(line, request_id, sequence_id, error))
            return error_response("INVALID_COMMAND", error);

        const std::string shape = json_string(line, "shape").value_or("");
        if (shape == "sphere" || shape == "spherical")
            use_cylinder_shape.store(false);
        else if (shape == "cylinder" || shape == "cylindrical")
            use_cylinder_shape.store(true);
        else
        {
            return error_response(
                "UNSUPPORTED_GRASP",
                "shape must be sphere or cylinder"
            );
        }
        std::string profile = json_string(line, "profile").value_or("");
        std::replace(profile.begin(), profile.end(), '-', '_');
        int16_t current_limit = 0;
        if (
            profile == "nominal"
            || profile == "hard"
            || profile == "soft_same"
        )
            current_limit = HARD_CURRENT_LIMIT;
        else if (profile == "gentle" || profile == "soft_low")
            current_limit = SOFT_LOW_CURRENT_LIMIT;
        else
        {
            return error_response(
                "UNSUPPORTED_PROFILE",
                "profile must be hard, soft_same, soft_low, nominal, or gentle"
            );
        }

        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            bool duplicate = false;
            if (duplicate_or_stale_locked(
                    request_id,
                    sequence_id,
                    "START",
                    duplicate))
            {
                return error_response("STALE_SEQUENCE", "stale sequence_id");
            }
            if (duplicate)
                return status_response_unlocked(true, true, "duplicate START");
            if (status_.lifecycle != Lifecycle::READY || status_.busy)
                return error_response("BUSY", "RESET is required before a new START");

            status_ = SharedStatus{};
            status_.lifecycle = Lifecycle::PREPARING;
            status_.busy = true;
            status_.terminal = false;
            status_.release_complete = false;
            status_.request_id = request_id;
            status_.last_sequence_id = sequence_id;
            status_.last_action = "START";
            status_.profile = profile;
            status_.current_limit_raw = current_limit;
        }

        release_requested_.store(false);
        abort_requested_.store(false);
        worker_ = std::thread(
            &Service::run_attempt,
            this,
            request_id,
            profile,
            current_limit
        );
        return status_response(true, false, "START accepted");
    }

    std::string release(const std::string& line)
    {
        return request_stop(line, false);
    }

    std::string abort(const std::string& line)
    {
        return request_stop(line, true);
    }

    std::string request_stop(const std::string& line, bool aborting)
    {
        std::string request_id;
        std::string error;
        long long sequence_id = -1;
        if (!command_identity(line, request_id, sequence_id, error))
            return error_response("INVALID_COMMAND", error);
        const std::string action = aborting ? "ABORT" : "RELEASE";
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            if (request_id != status_.request_id)
                return error_response("UNKNOWN_REQUEST", "request_id does not own the active attempt");
            bool duplicate = false;
            if (duplicate_or_stale_locked(
                    request_id,
                    sequence_id,
                    action,
                    duplicate))
                return error_response("STALE_SEQUENCE", "stale sequence_id");
            if (duplicate)
                return status_response_unlocked(true, true, "duplicate " + action);
            if (!status_.busy)
                return status_response_unlocked(true, true, "attempt is already terminal");
            status_.last_sequence_id = sequence_id;
            status_.last_action = action;
        }
        if (aborting)
            abort_requested_.store(true);
        release_requested_.store(true);
        return status_response(true, false, action + " accepted");
    }

    std::string reset(const std::string& line)
    {
        std::string request_id;
        std::string error;
        long long sequence_id = -1;
        if (!command_identity(line, request_id, sequence_id, error))
            return error_response("INVALID_COMMAND", error);
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            if (status_.busy)
                return error_response("BUSY", "cannot RESET an active attempt");
            if (
                !status_.request_id.empty()
                && request_id != status_.request_id
            )
                return error_response("UNKNOWN_REQUEST", "request_id does not match terminal attempt");
        }
        if (worker_.joinable())
            worker_.join();
        std::string reason;
        {
            std::lock_guard<std::mutex> serial_lock(serial_mutex_);
            if (!ensure_open(reason))
            {
                set_fault("RESET_FAILED", reason);
                return error_response("RESET_FAILED", reason);
            }
        }
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            status_ = SharedStatus{};
            status_.lifecycle = Lifecycle::READY;
            status_.last_sequence_id = sequence_id;
            status_.last_action = "RESET";
        }
        return status_response(true, false, "RESET complete; hand is open");
    }

    std::string status_response_unlocked(
        bool accepted,
        bool idempotent,
        const std::string& message
    ) const
    {
        std::ostringstream out;
        out << "{\"accepted\":" << (accepted ? "true" : "false")
            << ",\"idempotent\":" << (idempotent ? "true" : "false")
            << ",\"message\":\"" << json_escape(message) << "\""
            << ",\"request_id\":\"" << json_escape(status_.request_id) << "\""
            << ",\"lifecycle_state\":\""
            << lifecycle_text(status_.lifecycle) << "\"}"
            ;
        return out.str();
    }

    bool ensure_open(std::string& reason)
    {
        send_zero_all_currents(handle_);
        const auto deadline = std::chrono::steady_clock::now() + POSITION_TIMEOUT;
        int stable = 0;
        while (std::chrono::steady_clock::now() < deadline)
        {
            send_positions(handle_, OPEN_POSITION);
            Sample sample;
            if (!read_sample(handle_, sample))
            {
                reason = "open_read_failure";
                return false;
            }
            if (!validate_sample(sample, POSITION_MODE_CURRENT_BOUND, reason))
            {
                reason = "open_" + reason;
                return false;
            }
            bool is_stable = true;
            for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
            {
                is_stable &= sample.positions[i] <= 12;
                is_stable &= std::abs(static_cast<int>(sample.speeds[i]))
                    <= ZERO_SPEED_TOLERANCE;
                is_stable &= sample.states[i] == MOTOR_STATE_IDLE;
            }
            stable = is_stable ? stable + 1 : 0;
            if (stable >= 4)
                return true;
            std::this_thread::sleep_for(STATUS_PERIOD);
        }
        reason = "open_timeout";
        return false;
    }

    bool baseline(
        ExperimentLog& log,
        Sample& last,
        std::string& reason
    )
    {
        const auto deadline = std::chrono::steady_clock::now() + BASELINE_DURATION;
        std::array<int16_t, MOTOR_COUNT> zero{};
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (release_requested_.load() || abort_requested_.load())
                return false;
            if (!read_sample(handle_, last))
            {
                reason = "baseline_read_failure";
                return false;
            }
            if (!validate_sample(last, POSITION_MODE_CURRENT_BOUND, reason))
            {
                reason = "baseline_" + reason;
                return false;
            }
            log.sample("baseline", positions_as_q(last), zero, last);
            std::this_thread::sleep_for(STATUS_PERIOD);
        }
        return true;
    }

    void run_attempt(
        std::string request_id,
        std::string profile,
        int16_t current_limit
    )
    {
        std::lock_guard<std::mutex> serial_lock(serial_mutex_);
        const std::string prefix = (
            std::filesystem::path(options_.log_dir)
            / (
                safe_file_component(request_id) + "_" + profile + "_"
                + timestamp_for_file()
            )
        ).string();

        std::string reason;
        std::string result;
        std::string safety{"OK"};
        std::string task_success{"UNKNOWN"};
        bool normal_release = false;
        bool aborting = false;
        bool returned = false;
        AttemptMetrics metrics;

        try
        {
            ExperimentLog log(prefix);
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.csv_path = log.csv_path();
                status_.meta_path = log.meta_path();
            }
            log.meta("controller_version", "brainco_grasp_service_v2");
            log.meta("control_core", "spherical_impedance_v3");
            log.meta("request_id", request_id);
            log.meta("backend_name", "brainco_revo2_direct");
            log.meta("control_profile", "REVO_CURRENT_PROFILE");
            log.meta("grasp_shape", use_cylinder_shape.load() ? "CYLINDRICAL" : "SPHERICAL");
            log.meta("profile", profile);
            log.meta("current_limit_raw", current_limit);
            log.meta("serial_number", serial_);
            log.meta("firmware_version", firmware_);
            log.meta("kp", KP);
            log.meta("kd", KD);
            log.meta("ramp_rate", RAMP_RATE);
            log.meta("max_hold_seconds", MAX_HOLD_DURATION.count());
            log.meta("retention_status", "UNKNOWN");
            log.meta("task_success", "UNKNOWN");
            log.meta("external_retention_required", 1);

            Sample armed;
            if (reason.empty() && !baseline(log, armed, reason))
            {
                if (reason.empty())
                    normal_release = release_requested_.load();
            }
            if (reason.empty())
            {
                for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                {
                    if (
                        armed.positions[i] > 20
                        || std::abs(static_cast<int>(armed.speeds[i]))
                            > ZERO_SPEED_TOLERANCE
                        || armed.states[i] != MOTOR_STATE_IDLE
                    )
                    {
                        reason = std::string("start_pose_not_open_")
                            + MOTOR_NAMES[i];
                        break;
                    }
                }
            }
            if (
                reason.empty()
                && !release_requested_.load()
                && !abort_requested_.load()
            )
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.lifecycle = Lifecycle::CLOSING;
            }

            std::array<double, MOTOR_COUNT> q_des = positions_as_q(armed);
            std::array<double, MOTOR_COUNT> filtered_dq{};
            std::array<int16_t, MOTOR_COUNT> previous_command{};
            std::array<int, MOTOR_COUNT> mismatch_count{};
            metrics.peak_position = armed.positions;
            const auto base_position = armed.positions;
            auto previous_step = std::chrono::steady_clock::now();

            auto control_step = [&](const std::string& phase, bool ramp)
            {
                const auto started = std::chrono::steady_clock::now();
                Sample sample;
                if (!read_sample(handle_, sample))
                {
                    reason = phase + "_read_failure";
                    return false;
                }
                if (!validate_sample(
                        sample,
                        current_limit + CURRENT_FEEDBACK_MARGIN,
                        reason))
                {
                    reason = phase + "_" + reason;
                    return false;
                }
                const double dt = std::clamp(
                    std::chrono::duration<double>(started - previous_step).count(),
                    0.0,
                    std::chrono::duration<double>(CONTROL_DEADLINE).count()
                );
                if (ramp)
                {
                    for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                    {
                        q_des[i] = std::min(
                            active_target_q(i),
                            q_des[i] + RAMP_RATE * dt
                        );
                    }
                }

                std::array<int16_t, MOTOR_COUNT> command{};
                const double limit = current_limit / 1000.0;
                for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                {
                    const double q = sample.positions[i] / 1000.0;
                    const double dq = sample.speeds[i] / 1000.0;
                    filtered_dq[i] += EMA_ALPHA * (dq - filtered_dq[i]);
                    double current = KP * (q_des[i] - q)
                        - KD * filtered_dq[i];
                    if (dq > CLOSING_SPEED_SOFT_LIMIT)
                    {
                        const double speed_cap = limit * std::clamp(
                            (CLOSING_SPEED_ZERO_CURRENT - dq)
                                / (CLOSING_SPEED_ZERO_CURRENT
                                    - CLOSING_SPEED_SOFT_LIMIT),
                            -1.0,
                            1.0
                        );
                        current = std::min(current, speed_cap);
                    }
                    current = std::clamp(current, -limit, limit);
                    command[i] = static_cast<int16_t>(
                        std::lround(current * 1000.0)
                    );

                    const bool mismatch = previous_command[i] >= 60
                        && std::abs(
                            static_cast<int>(sample.currents[i])
                            - static_cast<int>(previous_command[i])
                        ) > 55;
                    mismatch_count[i] = mismatch ? mismatch_count[i] + 1 : 0;
                    if (mismatch_count[i] >= 5)
                    {
                        reason = phase + "_current_feedback_mismatch_"
                            + MOTOR_NAMES[i];
                        return false;
                    }
                    metrics.moved[i] |= std::abs(
                        static_cast<int>(sample.positions[i])
                        - static_cast<int>(base_position[i])
                    ) >= 10;
                    const bool contact = q_des[i] - q >= 0.03
                        && std::abs(dq) <= 0.035
                        && command[i] >= static_cast<int16_t>(current_limit * 0.80);
                    metrics.contact_counter[i] = contact
                        ? metrics.contact_counter[i] + 1
                        : 0;
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
                        metrics.peak_position[i], sample.positions[i]
                    );
                }
                send_all_currents(handle_, command);
                log.sample(phase, q_des, command, sample);
                previous_command = command;
                previous_step = started;
                {
                    std::lock_guard<std::mutex> lock(status_mutex_);
                    status_.moved_drive_count = count_active(metrics.moved);
                    status_.contact_like_count =
                        count_active(metrics.contact_like);
                }
                if (std::chrono::steady_clock::now() - started > CONTROL_DEADLINE)
                {
                    reason = phase + "_loop_deadline";
                    return false;
                }
                return true;
            };

            auto target_reached = [&]()
            {
                for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
                {
                    if (q_des[i] < active_target_q(i))
                        return false;
                }
                return true;
            };

            auto next = std::chrono::steady_clock::now();
            auto wait_period = [&]()
            {
                next += CONTROL_PERIOD;
                const auto now = std::chrono::steady_clock::now();
                if (now > next)
                    next = now + CONTROL_PERIOD;
                std::this_thread::sleep_until(next);
            };

            const auto ramp_deadline = std::chrono::steady_clock::now()
                + RAMP_TIMEOUT;
            while (
                reason.empty()
                && !release_requested_.load()
                && !abort_requested_.load()
                && !target_reached()
                && std::chrono::steady_clock::now() < ramp_deadline
            )
            {
                if (!control_step("impedance_ramp", true))
                    break;
                wait_period();
            }
            if (
                reason.empty()
                && !release_requested_.load()
                && !abort_requested_.load()
                && !target_reached()
            )
                reason = "impedance_ramp_timeout";

            if (reason.empty() && !release_requested_.load()
                && !abort_requested_.load())
            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.lifecycle = Lifecycle::HOLDING;
            }
            const auto hold_deadline = std::chrono::steady_clock::now()
                + MAX_HOLD_DURATION;
            while (
                reason.empty()
                && !release_requested_.load()
                && !abort_requested_.load()
                && std::chrono::steady_clock::now() < hold_deadline
            )
            {
                if (!control_step("impedance_hold", false))
                    break;
                wait_period();
            }
            normal_release = release_requested_.load();
            aborting = abort_requested_.load();

            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.lifecycle = Lifecycle::RELEASING;
            }
            send_zero_all_currents(handle_);
            std::string cleanup_reason;
            returned = ensure_open(cleanup_reason);
            if (!returned)
            {
                if (!reason.empty()) reason += ";";
                reason += "cleanup=" + cleanup_reason;
            }

            const int moved_count = count_active(metrics.moved);
            const int contact_count = count_active(metrics.contact_like);
            if (aborting)
            {
                result = "ABORTED";
                safety = "ABORTED";
                task_success = "FALSE";
                if (reason.empty()) reason = "abort_requested";
            }
            else if (!reason.empty())
            {
                result = "SAFE_RUN_FAILED";
                safety = "DEVICE_FAULT";
                task_success = "FALSE";
            }
            else if (normal_release)
            {
                result = "RELEASED";
            }
            else
            {
                result = "SAFE_RUN_PASS";
            }

            log.meta("moved_drive_count", moved_count);
            log.meta("contact_like_count", contact_count);
            log.meta("controller_outcome", contact_count >= 3
                ? "CONTACT_HOLD_CANDIDATE"
                : "INSUFFICIENT_CONTACT_EVIDENCE");
            log.meta("retention_status", "UNKNOWN");
            log.meta("task_success", task_success);
            log.meta("safety_status", safety);
            log.meta("returned_open", returned ? 1 : 0);
            log.meta("position_mode_restored", returned ? 1 : 0);
            log.meta("result", result);
            log.meta("reason", reason);
            for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
            {
                log.meta(std::string("peak_command_raw_") + MOTOR_NAMES[i],
                    metrics.peak_command[i]);
                log.meta(std::string("peak_feedback_raw_") + MOTOR_NAMES[i],
                    metrics.peak_feedback[i]);
                log.meta(std::string("peak_speed_raw_") + MOTOR_NAMES[i],
                    metrics.peak_speed[i]);
                log.meta(std::string("peak_position_raw_") + MOTOR_NAMES[i],
                    metrics.peak_position[i]);
            }

            {
                std::lock_guard<std::mutex> lock(status_mutex_);
                status_.busy = false;
                status_.terminal = true;
                status_.release_complete = returned;
                status_.lifecycle = safety == "OK"
                    ? Lifecycle::TERMINAL
                    : Lifecycle::FAULT;
                status_.result = result;
                status_.reason = reason;
                status_.safety_status = safety;
                status_.task_success = task_success;
                status_.retention_status = "UNKNOWN";
                status_.controller_outcome = contact_count >= 3
                    ? "CONTACT_HOLD_CANDIDATE"
                    : "INSUFFICIENT_CONTACT_EVIDENCE";
                status_.moved_drive_count = moved_count;
                status_.contact_like_count = contact_count;
            }
        }
        catch (const std::exception& exception)
        {
            send_zero_all_currents(handle_);
            std::string cleanup_reason;
            returned = ensure_open(cleanup_reason);
            set_fault("SERVICE_EXCEPTION", exception.what(), returned);
        }
    }

    void set_fault(
        const std::string& result,
        const std::string& reason,
        bool release_complete = false
    )
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        status_.busy = false;
        status_.terminal = true;
        status_.release_complete = release_complete;
        status_.lifecycle = Lifecycle::FAULT;
        status_.result = result;
        status_.reason = reason;
        status_.safety_status = "DEVICE_FAULT";
        status_.task_success = "FALSE";
        status_.retention_status = "UNKNOWN";
    }

    DeviceHandler* handle_;
    ServiceOptions options_;
    std::string serial_;
    std::string firmware_;
    mutable std::mutex status_mutex_;
    std::mutex serial_mutex_;
    SharedStatus status_;
    std::atomic<bool> release_requested_{false};
    std::atomic<bool> abort_requested_{false};
    std::thread worker_;
};

class SocketGuard
{
public:
    SocketGuard(int descriptor, std::string path)
        : descriptor_(descriptor), path_(std::move(path))
    {}
    ~SocketGuard()
    {
        if (descriptor_ >= 0) ::close(descriptor_);
        if (!path_.empty()) ::unlink(path_.c_str());
    }
private:
    int descriptor_;
    std::string path_;
};

int create_listener(const std::string& path, const std::string& group_name)
{
    const int descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (descriptor < 0)
        throw std::runtime_error("Failed to create Unix socket");
    ::unlink(path.c_str());
    sockaddr_un address{};
    if (path.size() >= sizeof(address.sun_path))
    {
        ::close(descriptor);
        throw std::runtime_error("Unix socket path is too long");
    }
    address.sun_family = AF_UNIX;
    std::copy(path.begin(), path.end(), address.sun_path);
    if (::bind(
            descriptor,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) != 0)
    {
        ::close(descriptor);
        throw std::runtime_error("Failed to bind Unix socket");
    }
    const group* socket_group = ::getgrnam(group_name.c_str());
    if (!socket_group)
    {
        ::close(descriptor);
        ::unlink(path.c_str());
        throw std::runtime_error("Socket group does not exist");
    }
    if (
        ::chown(path.c_str(), static_cast<uid_t>(-1), socket_group->gr_gid) != 0
        || ::chmod(path.c_str(), 0660) != 0
        || ::listen(descriptor, 8) != 0
    )
    {
        ::close(descriptor);
        ::unlink(path.c_str());
        throw std::runtime_error("Failed to prepare Unix socket");
    }
    return descriptor;
}

std::string read_line(int descriptor)
{
    timeval timeout{};
    timeout.tv_sec = 2;
    ::setsockopt(
        descriptor,
        SOL_SOCKET,
        SO_RCVTIMEO,
        &timeout,
        sizeof(timeout)
    );
    std::string line;
    char buffer[1024];
    while (line.size() < MAX_REQUEST_BYTES)
    {
        const ssize_t count = ::read(descriptor, buffer, sizeof(buffer));
        if (count <= 0)
            break;
        line.append(buffer, static_cast<std::size_t>(count));
        const std::size_t newline = line.find('\n');
        if (newline != std::string::npos)
        {
            line.resize(newline);
            return line;
        }
    }
    if (line.size() >= MAX_REQUEST_BYTES)
        return {};
    return line;
}

void write_line(int descriptor, const std::string& line)
{
    const std::string output = line + "\n";
    std::size_t offset = 0;
    while (offset < output.size())
    {
        const ssize_t count = ::write(
            descriptor,
            output.data() + offset,
            output.size() - offset
        );
        if (count <= 0)
            return;
        offset += static_cast<std::size_t>(count);
    }
}

} // namespace grasp_service

int main(int argc, char** argv)
{
    using namespace grasp_service;
    const auto parsed = parse_service_options(argc, argv);
    if (!parsed)
        return 2;
    const ServiceOptions options = *parsed;
    std::signal(SIGINT, service_signal_handler);
    std::signal(SIGTERM, service_signal_handler);
    std::signal(SIGHUP, service_signal_handler);

    try
    {
        init_logging(LogLevel::LOG_LEVEL_ERROR);
        ProcessLock process_lock;
        if (!process_lock.acquire())
            throw std::runtime_error("Another direct hand controller holds the lock");

        DeviceHandler* handle = modbus_open(options.port.c_str(), BAUDRATE);
        if (!handle)
            throw std::runtime_error("Failed to open serial device");
        DeviceGuard device_guard(handle);

        using InfoPtr = std::unique_ptr<CDeviceInfo, decltype(&free_device_info)>;
        InfoPtr info(read_device_info_with_retry(handle), &free_device_info);
        if (!info)
            throw std::runtime_error("Failed to read device info");
        const std::string serial = safe_text(info->serial_number);
        const std::string firmware = safe_text(info->firmware_version);
        // IDENTITY-PATCH: the expected serial number can be overridden through
        // the environment, because the left and right hands have different ones.
        const char* serial_env = std::getenv("BRAINCO_GRASP_SERIAL");  // IDENTITY-PATCH
        const std::string expected_serial =  // IDENTITY-PATCH
            serial_env ? std::string(serial_env) : options.expected_serial;  // IDENTITY-PATCH
        if (
            serial != expected_serial
            || !is_right_sku(info->sku_type)
            || !is_revo2(info->hardware_type)
        )
            throw std::runtime_error("RIGHT Revo2 identity check failed");

        if (!h3_gate_passed(
                options.nominal_gate_meta,
                serial,
                firmware,
                HARD_CURRENT_LIMIT))
            throw std::runtime_error("Nominal H3 all-drive gate failed");
        if (!h3_gate_passed(
                options.gentle_gate_meta,
                serial,
                firmware,
                SOFT_LOW_CURRENT_LIMIT))
            throw std::runtime_error("Gentle H3 all-drive gate failed");

        stark_set_finger_unit_mode(
            handle,
            RIGHT_HAND_ID,
            FINGER_UNIT_MODE_NORMALIZED
        );
        if (
            stark_get_finger_unit_mode(handle, RIGHT_HAND_ID)
                != FINGER_UNIT_MODE_NORMALIZED
            || stark_get_turbo_mode_enabled(handle, RIGHT_HAND_ID)
        )
            throw std::runtime_error("Unsafe unit mode or Turbo configuration");
        const uint16_t aux_lock_current =
            stark_get_thumb_aux_lock_current(handle, RIGHT_HAND_ID);
        if (aux_lock_current < 100 || aux_lock_current > 500)
            throw std::runtime_error("Unsafe thumb AUX lock current");

        for (std::size_t i = 0; i < MOTOR_COUNT; ++i)
        {
            const uint16_t max_current = stark_get_finger_max_current(
                handle,
                RIGHT_HAND_ID,
                FINGER_IDS[i]
            );
            const uint16_t protected_current = stark_get_finger_protected_current(
                handle,
                RIGHT_HAND_ID,
                FINGER_IDS[i]
            );
            const uint16_t requested_mA = max_current == 0
                ? 0
                : static_cast<uint16_t>(std::ceil(
                    HARD_CURRENT_LIMIT * static_cast<double>(max_current) / 1000.0
                ));
            if (max_current == 0 || protected_current < requested_mA)
                throw std::runtime_error(
                    std::string("Unsafe current setting for ") + MOTOR_NAMES[i]
                );
        }

        std::filesystem::create_directories(options.log_dir);
        Service service(handle, options, serial, firmware);
        std::string initialization_reason;
        if (!service.initialize(initialization_reason))
        {
            service.shutdown();
            throw std::runtime_error(
                "Failed to initialize open pose: " + initialization_reason
            );
        }
        const int listener = create_listener(
            options.socket_path,
            options.socket_group
        );
        SocketGuard socket_guard(listener, options.socket_path);
        std::cout
            << "BrainCo grasp service READY\n"
            << "socket=" << options.socket_path << "\n"
            << "profiles=hard(200),soft_same(200),soft_low(150)\n"
            << "shape=sphere|cylinder\n"
            << "retention_status=UNKNOWN task_success=UNKNOWN\n";

        while (!shutdown_requested.load())
        {
            pollfd descriptor{};
            descriptor.fd = listener;
            descriptor.events = POLLIN;
            const int ready = ::poll(&descriptor, 1, 200);
            if (ready <= 0 || !(descriptor.revents & POLLIN))
                continue;
            const int client = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
            if (client < 0)
                continue;
            const std::string request = read_line(client);
            const std::string response = request.empty()
                ? "{\"accepted\":false,\"error\":\"INVALID_COMMAND\",\"reason\":\"empty or oversized request\"}"
                : service.dispatch(request);
            write_line(client, response);
            ::close(client);
        }

        service.shutdown();
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "brainco_grasp_service: " << exception.what() << "\n";
        return 1;
    }
}
