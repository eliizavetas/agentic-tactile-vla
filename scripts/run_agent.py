#!/usr/bin/env python3
"""Entry point on the robot: the VLA bridge with the execution supervisor.

Run inside the GR00T-WholeBodyControl data-collection environment, after the
policy server, camera, hand service, whole-body controller and keyboard
publisher are up (see docs/deployment.md):

    HAND_ENABLE=1 python scripts/run_agent.py --prompt "pick up the object and place it down"

    # base VLA, supervisor off
    HAND_ENABLE=1 python scripts/run_agent.py --no-agent --prompt "..."

Connection settings come from the environment (see .env.example) and can be
overridden with the bridge's own flags, which are passed through unchanged.

The console output is also written to $AGENT_LOG_DIR/bridge_<hand>_<time>.log.
The supervisor's verdicts_* and ramp_* files of the same run carry the same
<time>, and a restart never overwrites an earlier session.
"""

import argparse
import os
import sys
import time


class _Tee:
    """Write everything to the console and to the session log file."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _start_session_log(hand: str) -> str:
    """Open the session log and return the session time stamp."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.environ["AGENT_SESSION_STAMP"] = stamp
    directory = os.environ.get("AGENT_LOG_DIR", "agent_logs")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"bridge_{hand}_{stamp}.log")
    log = open(path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    print(f"[session] console log: {path}", flush=True)
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-agent", action="store_true",
                        help="run the base VLA without the supervisor")
    parser.add_argument("--hand", default=os.environ.get("AGENT_HAND", "left"),
                        choices=["left", "right"])
    parser.add_argument("--variant", default=os.environ.get("AGENT_VARIANT", "S-full"),
                        choices=["S-full", "S-position", "S-retain", "U-full", "U-stall"])
    parser.add_argument("--observe", action="store_true",
                        help="run and log the supervisor without overriding anything")
    parser.add_argument("--vlm", action="store_true",
                        help="use the VLM service for mode selection")
    parser.add_argument("--handling-cue", default=None,
                        help="handling text for the interaction parser only, e.g. 'gently'")
    args, bridge_args = parser.parse_known_args()

    os.environ["AGENT_ENABLE"] = "0" if args.no_agent else "1"
    os.environ["AGENT_HAND"] = args.hand
    os.environ["AGENT_VARIANT"] = args.variant
    if args.observe:
        os.environ["AGENT_OBSERVE"] = "1"
    if args.vlm:
        os.environ["VLM_ENABLE"] = "1"
    if args.handling_cue:
        os.environ["AGENT_HANDLING_CUE"] = args.handling_cue
    _start_session_log(args.hand)

    import tyro

    from agentic_tactile_vla.bridge import vla_bridge

    sys.argv = [sys.argv[0]] + bridge_args
    vla_bridge.main(tyro.cli(vla_bridge.InferenceConfig))


if __name__ == "__main__":
    main()
