"""Client for the GR00T PolicyServer that is robust to msgpack_numpy.patch().

Some packages in the robot environment call msgpack_numpy.patch() at import
time, which changes how numpy arrays are packed and breaks the stock client.
This client converts every array to the server's own wire format before
packing, so the result does not depend on that global patch.
"""
from __future__ import annotations
import io
from typing import Any
import msgpack
import numpy as np
import zmq


def _encode(obj: Any) -> Any:
    """Recursively convert ndarrays to the server format BEFORE packing."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if b"__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj[b"as_npy"]), allow_pickle=False)
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _pack(data: Any) -> bytes:
    return msgpack.packb(_encode(data), use_bin_type=True)


def _unpack(raw: bytes) -> Any:
    return _decode(msgpack.unpackb(raw, raw=False, strict_map_key=False))


class Gr00tClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context()
        self._connect()

    def _connect(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self.socket.send(_pack(req))
            msg = self.socket.recv()
        except zmq.error.Again:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self._connect()
            raise RuntimeError(f"GR00T server timeout on '{endpoint}'")
        resp = _unpack(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def get_modality_config(self) -> Any:
        return self.call("get_modality_config", requires_input=False)

    def get_action(self, observation: dict, options: dict | None = None):
        resp = self.call("get_action", {"observation": observation, "options": options})
        if not isinstance(resp, (list, tuple)) or len(resp) != 2:
            raise RuntimeError(f"Unexpected get_action response: {type(resp)!r}")
        action, info = tuple(resp)
        return action, info if isinstance(info, dict) else {}

    def close(self):
        try:
            self.socket.close(); self.context.term()
        except Exception:
            pass
