# Implements the paper specification. Not re-validated on hardware since it was written.
"""Client for the mode-selection VLM service (scripts/vlm_service.py).

One request carries a JPEG frame and the fixed mode-selection prompt. The
service returns the raw model text. Parsing happens on the robot side in
`modes.parse_vlm_reply`, which accepts only the two mode labels.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Optional


class VLMClient:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 timeout: float = 5.0) -> None:
        self.host = host or os.environ.get("VLM_HOST", "localhost")
        self.port = int(port or os.environ.get("VLM_PORT", "5599"))
        self.timeout = timeout

    @staticmethod
    def encode_jpeg(image) -> bytes:
        """Encode an RGB uint8 frame as JPEG."""
        import cv2

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def query(self, image, prompt: str) -> str:
        """Blocking request; raises on timeout. Called from a worker thread."""
        import zmq

        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        socket.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        try:
            socket.connect(f"tcp://{self.host}:{self.port}")
            payload = {
                "image_jpeg_b64": base64.b64encode(self.encode_jpeg(image)).decode(),
                "prompt": prompt,
            }
            socket.send(json.dumps(payload).encode())
            reply = json.loads(socket.recv().decode())
        finally:
            socket.close()
        return reply.get("text", "")
