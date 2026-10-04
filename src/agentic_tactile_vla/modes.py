# Implements the paper specification. Not re-validated on hardware since it was written.
"""Interaction-mode selection (paper Sec. III-E).

Two predefined modes exist: nominal position control and compliant control.

* Explicit handling cues in the instruction ("gently", "carefully", "fragile")
  activate compliant mode.
* With no explicit cue, a vision-language model (Qwen2.5-VL-7B-Instruct)
  selects between the two modes from the image and the task context. It runs
  asynchronously while the hand is open, and one decision is cached per grasp
  episode.
* Language cues take priority. Missing, late or malformed VLM replies default
  to nominal control.
* The VLM selects ONLY the mode. It does not estimate material properties and
  does not generate controller parameters. The reply parser enforces this: the
  only accepted replies are the two mode labels.
* Selection happens BEFORE closure, because a fragile object may deform before
  proprioceptive resistance becomes detectable.

The VLM never sees the handling cues: it receives a fixed image-and-task prompt
built from the task instruction with the cue words removed.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .config import ModeConfig


class InteractionMode(str, Enum):
    NOMINAL = "nominal"
    COMPLIANT = "compliant"


VLM_PROMPT = (
    "You are selecting how a robot hand should close on the object in the image.\n"
    "Task: {task}\n"
    "Answer with exactly one word and nothing else:\n"
    "compliant - if the object is fragile or thin-walled and a normal grasp could damage it\n"
    "nominal - otherwise"
)


def has_handling_cue(instruction: Optional[str], cfg: ModeConfig | None = None) -> bool:
    """True when the instruction contains an explicit handling cue."""
    cfg = cfg or ModeConfig()
    if not instruction:
        return False
    words = re.findall(r"[a-z]+", instruction.lower())
    return any(cue in words for cue in cfg.handling_cues)


def strip_handling_cues(instruction: Optional[str], cfg: ModeConfig | None = None) -> str:
    """Task text without handling cues, as passed to the VLM."""
    cfg = cfg or ModeConfig()
    if not instruction:
        return ""
    pattern = r"\b(" + "|".join(map(re.escape, cfg.handling_cues)) + r")\b"
    text = re.sub(pattern, " ", instruction, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def parse_vlm_reply(reply: object) -> Optional[InteractionMode]:
    """Strict parser for a VLM reply.

    Accepts exactly one of the two mode labels, ignoring surrounding whitespace
    and letter case. Anything else (extra words, punctuation, JSON, numbers,
    material descriptions, parameter values, an empty or non-text reply) is
    malformed and returns None, which the selector maps to nominal control.
    """
    if not isinstance(reply, str):
        return None
    label = reply.strip().lower()
    for mode in InteractionMode:
        if label == mode.value:
            return mode
    return None


@dataclass
class ModeDecision:
    mode: InteractionMode
    source: str                 # "language", "vlm" or "default"
    vlm_reply: Optional[str] = None
    detail: str = ""


class ModeSelector:
    """Chooses the interaction mode for one grasp episode.

    `vlm_query(image, prompt) -> str` performs one blocking VLM request and
    returns the raw reply text. It is called from a background thread, so the
    control loop never waits for it.
    """

    def __init__(self, vlm_query: Optional[Callable[[object, str], str]] = None,
                 cfg: ModeConfig | None = None) -> None:
        self.cfg = cfg or ModeConfig()
        self._vlm_query = vlm_query
        self._lock = threading.Lock()
        self._episode = 0
        self._requested = False
        self._reply: Optional[str] = None
        self._reply_ready = False
        self._decision: Optional[ModeDecision] = None

    # -- episode bookkeeping ----------------------------------------------------
    def new_episode(self) -> None:
        """Start a new grasp episode (called once the hand has reopened)."""
        with self._lock:
            self._episode += 1
            self._requested = False
            self._reply = None
            self._reply_ready = False
            self._decision = None

    # -- asynchronous VLM request -------------------------------------------------
    def maybe_request(self, image: object, task_instruction: Optional[str]) -> bool:
        """Launch the VLM request for this episode, once, while the hand is open.

        No request is made when the instruction carries an explicit cue (the
        cue decides), when no VLM is configured or when no image is available.
        """
        if self._vlm_query is None or image is None:
            return False
        if has_handling_cue(task_instruction, self.cfg):
            return False
        with self._lock:
            if self._requested:
                return False
            self._requested = True
            episode = self._episode
        prompt = VLM_PROMPT.format(task=strip_handling_cues(task_instruction, self.cfg))
        thread = threading.Thread(target=self._worker, args=(episode, image, prompt), daemon=True)
        thread.start()
        return True

    def _worker(self, episode: int, image: object, prompt: str) -> None:
        try:
            reply = self._vlm_query(image, prompt)
        except Exception:  # a failed request is a missing reply
            reply = None
        with self._lock:
            if episode == self._episode:   # a reply for an old episode is discarded
                self._reply = reply if isinstance(reply, str) else None
                self._reply_ready = True

    # -- decision ---------------------------------------------------------------
    def decide(self, instruction: Optional[str]) -> ModeDecision:
        """Fix the mode for this episode. Call when the closure is armed.

        A VLM reply that has not arrived by now is late and is ignored: the
        decision is cached, so a later reply cannot change it mid-closure.
        """
        with self._lock:
            if self._decision is not None:
                return self._decision
            if has_handling_cue(instruction, self.cfg):
                decision = ModeDecision(InteractionMode.COMPLIANT, "language")
            elif self._reply_ready:
                parsed = parse_vlm_reply(self._reply)
                if parsed is None:
                    decision = ModeDecision(InteractionMode.NOMINAL, "default",
                                            self._reply, "malformed or missing VLM reply")
                else:
                    decision = ModeDecision(parsed, "vlm", self._reply)
            else:
                detail = "VLM reply late" if self._requested else "no VLM request"
                decision = ModeDecision(InteractionMode.NOMINAL, "default", None, detail)
            self._decision = decision
            return decision
