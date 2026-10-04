#!/usr/bin/env python3
"""Publish operator key presses to the bridge over ZMQ.

    k  start / stop the whole-body control loop
    i  move to the initial pose and switch to POSE mode (required before 'p')
    p  pause / resume the policy loop
    g  freeze / resume body-reference progression by hand

Type a key and press Enter. A line starting with "prompt:" changes the
instruction, for example:  prompt:pick up the object and place it down
"""

import os
import time

import zmq


def main() -> None:
    port = int(os.environ.get("KEYBOARD_PORT", "5580"))
    pub = zmq.Context().socket(zmq.PUB)
    pub.bind(f"tcp://localhost:{port}")
    time.sleep(0.5)
    print("keys: k / i / p / g   (Ctrl+C to quit)")
    while True:
        key = input().strip()
        if key:
            pub.send_string(key)
            print("sent:", key)


if __name__ == "__main__":
    main()
