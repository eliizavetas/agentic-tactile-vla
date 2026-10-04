#!/usr/bin/env python3
"""Mode-selection VLM service (runs on the GPU server).

Serves Qwen2.5-VL-7B-Instruct behind a ZMQ REP socket. The robot sends one
camera frame and the fixed mode-selection prompt while the hand is open. The
service returns the raw model text and nothing else:

    request   {"image_jpeg_b64": "...", "prompt": "..."}
    reply     {"text": "<model output>", "latency_s": 0.41}

The model is asked for exactly one word, "nominal" or "compliant". It selects
only the interaction mode. It is not asked for material properties or
controller parameters, and the robot-side parser rejects every reply that is
not one of the two labels (a rejected reply defaults to nominal control).

    python scripts/vlm_service.py --port 5599
    python scripts/vlm_service.py --backend mock      # no model, for wiring tests
"""

import argparse
import base64
import io
import json
import os
import time


class MockBackend:
    """No model. Always answers "nominal". For connectivity tests only."""

    def generate(self, image_bytes: bytes, prompt: str) -> str:
        return "nominal"


class QwenBackend:
    def __init__(self, model_name: str, dtype: str = "bfloat16", max_new_tokens: int = 4):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        print(f"[vlm] loading {model_name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.model.eval()
        print("[vlm] model ready", flush=True)

    def generate(self, image_bytes: bytes, prompt: str) -> str:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["qwen", "mock"], default="qwen")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VLM_PORT", "5599")))
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()

    backend = MockBackend() if args.backend == "mock" else QwenBackend(args.model)

    import zmq

    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://{args.bind}:{args.port}")
    print(f"[vlm] serving on tcp://{args.bind}:{args.port} (backend {args.backend})", flush=True)
    while True:
        raw = socket.recv()
        t0 = time.time()
        try:
            request = json.loads(raw.decode())
            image = base64.b64decode(request["image_jpeg_b64"])
            text = backend.generate(image, request["prompt"])
            reply = {"text": text, "latency_s": round(time.time() - t0, 3)}
        except Exception as e:  # the robot treats an error as a missing reply
            reply = {"text": "", "error": f"{type(e).__name__}: {e}"}
        print(f"[vlm] {reply}", flush=True)
        socket.send(json.dumps(reply).encode())


if __name__ == "__main__":
    main()
