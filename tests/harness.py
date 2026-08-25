# SPDX-License-Identifier: Apache-2.0
"""Deterministic, GPU-free harness for observing scheduler decisions.

Built on vLLM's own test helpers (`tests/v1/core/utils.py`), which ship inside
the serving image. They construct a real `Scheduler`/`AsyncScheduler` with a
working KV cache manager, so these tests exercise the actual scheduling code
rather than a reimplementation of it.

What this measures is **per-step token allocation** -- `num_scheduled_tokens`
per request per step. That is exactly the quantity fair-share changes. Latency
is deliberately out of scope here; wall-clock belongs to the real-traffic runs.

## Scale

The helpers' stand-in model is `facebook/opt-125m`, whose context is **2048
tokens**, and that ceiling wins over any `max_model_len` argument. Prompts must
fit inside it.

This is not cosmetic. A prompt longer than the context makes
`max_model_len - num_computed_tokens - num_sampled_tokens_per_step` go negative,
`num_new_tokens` becomes 0, and the scheduler silently stops scheduling --
no error, no warning, just an empty `SchedulerOutput` forever. An earlier
attempt at this harness used 60k-token prompts and looked like a scheduler bug
for some time. Budgets and prompt sizes here are scaled down accordingly; the
allocation logic is scale-invariant, so the conclusions carry.
"""

import sys

# The helpers import themselves as `tests.v1...`, so the package root goes on
# the path, not the tests directory.
_VLLM_ROOT = "/workspace/vllm"
if _VLLM_ROOT not in sys.path:
    sys.path.insert(0, _VLLM_ROOT)

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

#: Per-step token budget for harness scenarios. Small so a prefill takes several
#: steps within the 2048-token context, making interleaving observable.
BUDGET = 256

#: Prompt length for "large" requests: several budgets' worth, still under 2048.
LARGE_PROMPT = 1600

CONTEXT_LIMIT = 2048


def build_scheduler(budget: int = BUDGET, **kwargs):
    """A real scheduler with a working KV cache manager and no GPU."""
    kwargs.setdefault("max_num_seqs", 64)
    kwargs.setdefault("enable_chunked_prefill", True)
    kwargs.setdefault("num_blocks", 20000)
    kwargs.setdefault("block_size", 16)
    return create_scheduler(max_num_batched_tokens=budget, **kwargs)


def make_requests(n: int, prompt_tokens: int = LARGE_PROMPT, max_tokens: int = 8):
    if prompt_tokens >= CONTEXT_LIMIT:
        raise ValueError(
            f"prompt_tokens={prompt_tokens} exceeds the harness model's "
            f"{CONTEXT_LIMIT}-token context; the scheduler will silently stall "
            f"rather than erroring. See this module's docstring."
        )
    return create_requests(
        num_requests=n, num_tokens=prompt_tokens, block_size=16, max_tokens=max_tokens
    )


def _model_output(sched, scheduler_output) -> ModelRunnerOutput:
    """Fake one model step.

    A request still mid-prefill must produce **no** sampled token. Handing one
    to a partially-prefilled request makes the scheduler treat it as decoding
    and stalls it -- a silent failure that broke two earlier attempts at this.
    """
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    sampled = []
    for rid in req_ids:
        req = sched.requests[rid]
        prefill_done = req.num_computed_tokens >= req.num_prompt_tokens
        sampled.append([1] if prefill_done else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def pending_prefills(sched) -> int:
    """Requests, running or waiting, that still have prompt tokens to process."""
    return sum(
        1
        for r in list(sched.running) + list(sched.waiting)
        if r.num_computed_tokens < r.num_prompt_tokens
    )


def drive(sched, steps: int, before_step=None) -> list[dict[str, int]]:
    """Run `steps` scheduler steps, returning per-step {request_id: tokens}.

    Stops early once the scheduler has nothing left to do, so callers can pass a
    generous step count. `before_step(sched)` runs immediately before each
    `schedule()` -- the hook fair-share policies use to set the threshold.
    """
    timeline: list[dict[str, int]] = []
    for _ in range(steps):
        if before_step is not None:
            before_step(sched)
        output = sched.schedule()
        if not output.num_scheduled_tokens:
            break
        timeline.append(dict(output.num_scheduled_tokens))
        sched.update_from_output(output, _model_output(sched, output))
    return timeline


def steps_with_all_progressing(timeline, req_ids) -> int:
    return sum(1 for step in timeline if all(step.get(r, 0) > 0 for r in req_ids))


def steps_until_prefill_done(timeline, req_id, prompt_tokens) -> int | None:
    """1-based step index at which `req_id` finishes its prefill, or None."""
    total = 0
    for i, step in enumerate(timeline, start=1):
        total += step.get(req_id, 0)
        if total >= prompt_tokens:
            return i
    return None
