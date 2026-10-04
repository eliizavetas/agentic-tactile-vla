#!/usr/bin/env python3

import argparse
import json
import socket
import sys
import time
import uuid
from typing import Optional


PROFILES = ("hard", "soft-same", "soft-low", "nominal", "gentle")


def send_request(socket_path: str, payload: dict) -> dict:
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3.0)
        client.connect(socket_path)
        client.sendall(encoded)
        response = bytearray()
        while b"\n" not in response:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > 65536:
                raise RuntimeError("service response is too large")
    if not response:
        raise RuntimeError("service returned no response")
    return json.loads(bytes(response).split(b"\n", 1)[0])


def show(response: dict) -> None:
    print(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True))


def next_sequence() -> int:
    return time.monotonic_ns()


def require_request_id(value: Optional[str]) -> str:
    if value:
        return value
    raise SystemExit("--request-id is required for this command")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thin client for the real BrainCo spherical grasp service"
    )
    parser.add_argument("--socket", default="/tmp/brainco_grasp.sock")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status")

    start = commands.add_parser("start")
    start.add_argument("shape", choices=("sphere", "cylinder"))
    start.add_argument("profile", choices=PROFILES)
    start.add_argument("--request-id")

    run = commands.add_parser("run")
    run.add_argument("shape", choices=("sphere", "cylinder"))
    run.add_argument("profile", choices=PROFILES)
    run.add_argument("--request-id")
    run.add_argument("--poll", type=float, default=0.25)
    run.add_argument(
        "--no-reset",
        action="store_true",
        help="leave the terminal attempt available for inspection",
    )

    for name in ("release", "abort", "reset"):
        command = commands.add_parser(name)
        command.add_argument("--request-id", required=True)

    args = parser.parse_args()
    if args.command == "status":
        show(send_request(args.socket, {"action": "STATUS"}))
        return 0

    request_id = getattr(args, "request_id", None)
    if args.command in ("start", "run"):
        request_id = request_id or f"grasp-{uuid.uuid4().hex[:12]}"
        response = send_request(
            args.socket,
            {
                "action": "START",
                "request_id": request_id,
                "sequence_id": next_sequence(),
                "shape": args.shape,
                "profile": args.profile,
            },
        )
        show(response)
        if not response.get("accepted") or args.command == "start":
            return 0 if response.get("accepted") else 2

        try:
            while True:
                time.sleep(max(0.05, args.poll))
                status = send_request(args.socket, {"action": "STATUS"})
                state = status.get("lifecycle_state", "UNKNOWN")
                print(
                    f"\r{state:10s} contacts={status.get('contact_like_count', 0)} "
                    f"result={status.get('result', ''):18s}",
                    end="",
                    flush=True,
                )
                if status.get("terminal"):
                    print()
                    show(status)
                    exit_code = 0 if status.get("safety_status") == "OK" else 3
                    if not args.no_reset and exit_code == 0:
                        reset_response = send_request(
                            args.socket,
                            {
                                "action": "RESET",
                                "request_id": request_id,
                                "sequence_id": next_sequence(),
                            },
                        )
                        print("RESET:")
                        show(reset_response)
                        if not reset_response.get("accepted"):
                            return 4
                    return exit_code
        except KeyboardInterrupt:
            print("\nRequesting RELEASE...")
            response = send_request(
                args.socket,
                {
                    "action": "RELEASE",
                    "request_id": request_id,
                    "sequence_id": next_sequence(),
                },
            )
            show(response)
            return 130 if response.get("accepted") else 4

    request_id = require_request_id(request_id)
    response = send_request(
        args.socket,
        {
            "action": args.command.upper(),
            "request_id": request_id,
            "sequence_id": next_sequence(),
        },
    )
    show(response)
    return 0 if response.get("accepted") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConnectionError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"brainco_grasp_cli: {exc}", file=sys.stderr)
        raise SystemExit(1)
