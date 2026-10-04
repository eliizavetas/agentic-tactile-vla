#!/usr/bin/env python3
"""
impedance_grasp.py — hand the gripper over to the compliant (current-impedance)
grasp controller and take it back.

WHY. The two control paths are complementary:
  * the position path with the supervisor answers "is the grasp supported?";
  * the current-limited controller answers "close without crushing".
The interaction-mode layer chooses between these two control modes before
closure.

WHAT THIS IS. A wrapper around `run_brainco_grasp_service.sh` and the
`brainco-grasp` CLI. All work happens in a BACKGROUND THREAD: the bridge's
loop runs at 50 Hz and must not block, while a handover takes seconds.

TRAPS ENCODED HERE (all learned on the real robot in August):
  1. The binary puts the TTY into RAW mode. If it inherits the terminal, input
     in the parent breaks. -> stdout/stderr to a FILE, stdin=DEVNULL,
     start_new_session=True, and `stty sane` after stopping it.
  2. It also block-buffers stdout into a pipe, so the READY line may never
     arrive. -> readiness is detected by POLLING THE STATUS SOCKET.
  3. Serial has one owner: run.sh refuses to start while brainco_hand_server
     is alive. -> kill it first, and ALWAYS bring it back afterwards.
  4. The relaunch used to fail silently because argv is relative
     (./brainco_hand_server). -> capture both argv and the working directory
     from /proc/<pid>/cwd.
  5. the controller's init requires an OPEN hand, otherwise
     `Failed to initialize open pose: open_position_bound_<finger>`.
     -> begin() is called when the closure is armed, while the hand is open.

PER-HAND SETTINGS are read from the environment and passed to the service:
  BRAINCO_GRASP_PORT_LEFT / _RIGHT      serial device of that hand
  BRAINCO_GRASP_SERIAL_LEFT / _RIGHT    serial number of that hand, as printed
                                        by brainco_hand_server at startup
  BRAINCO_GRASP_NOMINAL_GATE, BRAINCO_GRASP_GENTLE_GATE   H3 gate meta files
  BRAINCO_GRASP_GATE_ANY_SERIAL=1       only when the gate files were recorded
                                        on the other hand of the same model
The hand address is fixed by the device: 0x7f right, 0x7e left.

SUDO. The launcher refuses to run without sudo, and sudo STRIPS the
environment — hence the variables are passed as assignments inside the sudo
command line. A background thread also cannot type a password, hence `sudo -n`
everywhere. Warm the ticket with `sudo -v` before starting the bridge, or
install the sudoers rule from docs/deployment.md, which keeps exactly the six
variables passed below and allows only the launcher, the CLI and the two pkill
calls, all by absolute path. Otherwise every handover fails silently into
FAILED.

NOT HANDLED HERE. MAX_HOLD_DURATION is a compile-time constant in
brainco_grasp_service.cpp (120 s). After that time the service opens the hand
by itself.

Usage from the bridge:
    grasp = ImpedanceGrasp(hand="left")
    grasp.begin(profile="gentle")     # non-blocking
    ...                               # each tick: read grasp.state
    grasp.handback()                  # return the hand, object still gripped
"""

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

# Directory with the built service, the launcher and the brainco-grasp CLI.
SERVICE_DIR = Path(os.environ.get("BRAINCO_GRASP_SERVICE_DIR",
                                  Path(__file__).resolve().parent))
LOG = os.environ.get("BRAINCO_GRASP_CLIENT_LOG", "/tmp/impedance_grasp.log")

# Device addresses are fixed by the hand; ports and serial numbers are site
# specific and come from the environment (see .env.example).
SLAVE_ID = {"right": "0x7f", "left": "0x7e"}


def hand_config(hand):
    side = hand.upper()
    port = os.environ.get(f"BRAINCO_GRASP_PORT_{side}")
    serial = os.environ.get(f"BRAINCO_GRASP_SERIAL_{side}")
    if not port or not serial:
        raise RuntimeError(
            f"set BRAINCO_GRASP_PORT_{side} and BRAINCO_GRASP_SERIAL_{side}")
    return {"port": port, "slave": SLAVE_ID[hand], "serial": serial}


READY_TIMEOUT = 25.0     # wait for lifecycle_state=READY
GRASP_TIMEOUT = 25.0     # the closure ramp is slow; RAMP_TIMEOUT in the service is 15 s
POLL = 0.4


class ImpedanceGrasp:
    """
    States: IDLE -> HANDOVER -> HOLDING -> RETURNING -> IDLE
                            \\-> FAILED
    Everything heavy runs in a background thread; the agent only reads .state.
    """

    def __init__(self, hand="left", verbose=True):
        self.hand = hand
        self.cfg = hand_config(hand)
        self.verbose = verbose
        self.state = "IDLE"
        self.error = None
        self.last_status = {}
        self._proc = None
        self._thread = None
        self._handback = threading.Event()
        self._server_argv = None
        self._server_cwd = None

    # ------------------------------------------------------------ helpers
    def _say(self, msg):
        if self.verbose:
            print(f"[impedance] {msg}", flush=True)

    def _cli(self, *args, timeout=10):
        """Call brainco-grasp. Returns (returncode, combined output)."""
        try:
            r = subprocess.run(
                ["sudo", "-n", str(SERVICE_DIR / "brainco-grasp"), *args],
                cwd=str(SERVICE_DIR), capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            return 1, "timeout"

    def _status(self):
        rc, out = self._cli("status", timeout=6)
        d = {}
        try:
            import json as _json
            _start, _end = out.find("{"), out.rfind("}")
            if _start >= 0 and _end > _start:
                for k, v in _json.loads(out[_start:_end + 1]).items():
                    d[k] = str(v)
        except Exception:  # noqa: BLE001
            pass
        if not d:
            for k, v in re.findall(r'"?(\w+)"?\s*[=:]\s*"?([\w\-\./]+)"?', out):
                d[k] = v
        self.last_status = d
        return d

    def _capture_hand_server(self):
        """Remember how and from where brainco_hand_server was started."""
        try:
            out = subprocess.run(["pgrep", "-a", "-f", "brainco_hand_server"],
                                 capture_output=True, text=True).stdout.strip()
            for line in out.splitlines():
                pid, _, cmd = line.partition(" ")
                if "brainco_hand_server" not in cmd or "pgrep" in cmd:
                    continue
                self._server_argv = shlex.split(cmd)
                try:
                    self._server_cwd = os.readlink(f"/proc/{int(pid)}/cwd")
                except OSError:
                    self._server_cwd = os.getcwd()
                self._say(f"captured launch: {cmd}  (cwd {self._server_cwd})")
                return True
        except Exception as e:  # noqa: BLE001
            self._say(f"could not capture hand server launch: {e}")
        self._say("[dbg] capture found no hand server process")
        return False

    def _kill_hand_server(self):
        _p = subprocess.run(["sudo", "-n", "pkill", "-f", "brainco_hand_server"],
                            capture_output=True, text=True)
        self._say(f"[dbg] pkill rc={_p.returncode} "
                  f"out={_p.stdout.strip()!r} err={_p.stderr.strip()!r}")
        for _i in range(25):
            r = subprocess.run(["pgrep", "-f", "brainco_hand_server"],
                               capture_output=True)
            if r.returncode != 0:
                self._say(f"[dbg] port free after {_i * 0.2:.1f}s")
                time.sleep(0.5)          # let the serial port settle
                return True
            time.sleep(0.2)
        _left = subprocess.run(["pgrep", "-a", "-f", "brainco_hand_server"],
                               capture_output=True, text=True).stdout.strip()
        self._say(f"[dbg] STILL ALIVE after 5s:\n{_left}")
        return False

    def _restore_hand_server(self):
        if not self._server_argv:
            self._say("!! cannot restore hand server — argv was not captured")
            return False
        subprocess.Popen(self._server_argv, cwd=self._server_cwd,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        for _ in range(30):
            time.sleep(0.3)
            if subprocess.run(["pgrep", "-f", "brainco_hand_server"],
                              capture_output=True).returncode == 0:
                self._say("hand server is back up")
                return True
        self._say("!! hand server did NOT come back — restart it manually")
        return False

    # ------------------------------------------------------------ worker
    def _worker(self, profile, shape):
        try:
            self.state = "HANDOVER"
            self._capture_hand_server()

            if not self._kill_hand_server():
                raise RuntimeError("could not free the serial port")

            logf = open(LOG, "w")
            self._proc = subprocess.Popen(
                ["sudo", "-n",
                 f"BRAINCO_GRASP_PORT={self.cfg['port']}",
                 f"BRAINCO_GRASP_SLAVE={self.cfg['slave']}",
                 f"BRAINCO_GRASP_SERIAL={self.cfg['serial']}",
                 *[f"{k}={os.environ[k]}" for k in (
                     "BRAINCO_GRASP_NOMINAL_GATE", "BRAINCO_GRASP_GENTLE_GATE",
                     "BRAINCO_GRASP_GATE_ANY_SERIAL") if k in os.environ],
                 "bash", str(SERVICE_DIR / "run_brainco_grasp_service.sh")],
                cwd=str(SERVICE_DIR),
                stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            self._say(f"service starting ({self.cfg['port']}, {self.cfg['slave']}), log {LOG}")

            t0 = time.time()
            while time.time() - t0 < READY_TIMEOUT:
                if self._status().get("lifecycle_state") == "READY":
                    break
                time.sleep(POLL)
            else:
                raise RuntimeError(f"no READY within {READY_TIMEOUT:.0f} s — see {LOG}")
            self._say("READY")

            rc, out = self._cli("start", shape, profile,
                                "--request-id", f"agent-{int(time.time())}")
            if rc != 0:
                raise RuntimeError(f"start rejected: {out.strip()[:200]}")

            t0 = time.time()
            while time.time() - t0 < GRASP_TIMEOUT:
                st = self._status().get("lifecycle_state", "")
                if st == "HOLDING":
                    break
                if st in ("TERMINAL", "FAULT"):
                    raise RuntimeError(f"cycle ended without a hold: {st}")
                time.sleep(POLL)
            else:
                raise RuntimeError("never reached HOLDING")

            self.state = "HOLDING"
            self._say("holding the object — the body may carry on")

            self._handback.wait()        # wait for the agent to ask for the hand back

            self.state = "RETURNING"
            self._say("returning the hand to the VLA (object stays gripped)")
            self._stop_service()
            self._restore_hand_server()
            self.state = "IDLE"

        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.state = "FAILED"
            self._say(f"!! {e}")
            try:
                self._stop_service()
            finally:
                self._restore_hand_server()

    def _stop_service(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=6)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        subprocess.run(["sudo", "-n", "pkill", "-f", "brainco_grasp_service"])
        subprocess.run(["stty", "sane"])   # it may have left the tty in raw mode
        self._proc = None

    # ------------------------------------------------------------ API
    def begin(self, profile="gentle", shape="sphere"):
        """
        Request a handover. IMPORTANT: the hand must ALREADY BE OPEN, otherwise
        init fails with open_position_bound_<finger>. The bridge calls this
        when the closure is armed, while the hand is still open.
        """
        if self.state == "FAILED":
            self._say(f"retrying after: {self.error}")
            self.state, self.error = "IDLE", None
        if self.state != "IDLE":
            self._say(f"busy ({self.state}), begin ignored")
            return False
        self._handback.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._worker, args=(profile, shape), daemon=True)
        self._thread.start()
        return True

    def handback(self):
        """Return the hand to the VLA. The object stays gripped."""
        self._handback.set()

    @property
    def busy(self):
        return self.state in ("HANDOVER", "HOLDING", "RETURNING")

    def abort(self):
        """Emergency: release, stop the service, restore the hand server."""
        try:
            self._cli("abort", timeout=6)
        finally:
            self._handback.set()
