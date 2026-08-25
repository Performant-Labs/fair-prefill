# Motivation

Why this project exists, and the evidence behind it.

## The workload

A small number of peers — in the motivating case exactly two — each carrying a very
large, independently-evolving context against a **single GPU**. Concretely: long-running
agentic coding sessions, each sending prompts in the tens of thousands to low hundreds
of thousands of tokens, indefinitely, as conversation history accumulates.

This shape matters, because it is *not* the shape most serving guidance assumes. The
common case in the literature is **many small requests contending with a few large
ones**, where the goal is to stop one huge request from blocking a queue of cheap ones.
Here every participant is large. Mitigations designed for the former actively misfire on
the latter (see [alternatives-considered.md](alternatives-considered.md)).

## The mechanism

vLLM V1's scheduler, each step:

1. Walks the **running** queue in FCFS order.
2. Draws from a single shared per-step token budget (`max_num_batched_tokens`).
3. Then walks the **waiting** queue with whatever budget remains.

Nothing caps how much of that budget an already-running request's prefill may claim, and
the running queue is never reordered or round-robined between steps. So a sufficiently
large prefill consumes the entire budget every step until it finishes — and by the time
the waiting-queue loop runs, the budget is already zero.

The second client is therefore not *deprioritized*. It is **not scheduled at all** until
the first request's prefill completes. With prompts in the 40k–100k+ range and a budget
of a few thousand tokens per step, that is dozens of steps of complete starvation.

### Observable signature

- Waiting-by-reason shows `capacity` (ordinary queueing), **not** `deferred` — the
  request never advances far enough to be turned away for a resource reason; it never
  reaches the check at all.
- Requests-running sits at approximately 1 despite a much higher configured
  `max_num_seqs`.
- KV cache utilization stays moderate — well short of exhaustion. **The GPU is not out
  of memory. It is out of per-step token budget.** This distinction is what rules out
  KV-sizing as the cause and points squarely at scheduling.

## Why raising the token budget is not sufficient

Increasing `max_num_batched_tokens` genuinely helps, and is the recommended first move
for anyone hitting this. It shortens how many steps a single prefill needs, which
shortens the window during which it monopolizes the scheduler.

But note carefully *what* it improves: it reduces the **probability** that a second
request arrives during a monopolization window. It does nothing whatsoever about what
happens **when one does**. And it is bounded — the budget competes with KV cache for
GPU memory, so on a memory-constrained single-GPU deployment there is a hard ceiling
above which the engine fails to allocate a full-length KV cache at startup.

### The decay finding

This distinction is not academic. In multi-hour measurement on the motivating workload:

- Immediately after raising the budget, the collision rate was low and the great
  majority of large requests were served promptly. Measured over roughly the first hour,
  it looked like a large, clean win.
- Over the following hours the collision rate rose severalfold and then **held steady at
  a substantially higher level** — not a transient spike, a new equilibrium. Most of the
  apparent gain was gone.

Two obvious explanations were checked against the data and **ruled out**: prompt sizes
were stable across the whole window (contexts were not simply growing), and KV cache
utilization stayed moderate throughout (no memory-pressure onset).

The remaining explanation is **cadence entrainment**. A collision makes both clients'
turns take a similarly long time. Two agentic clients doing comparable work between
calls therefore tend to emerge from a collision closer together in phase than they went
in — so having collided once, they are *more* likely to collide again. The system drifts
into lock-step rather than naturally desynchronizing.

**The consequence for this project:** any fix that only reduces collision *frequency*
degrades over long sessions, because frequency is exactly what entrainment drives back
up. A durable fix has to reduce collision *cost* — what actually happens when two large
prefills genuinely overlap. That is what fair-share scheduling targets, and it is why
[measurement.md](measurement.md) makes TTFT-conditioned-on-collision the headline metric
rather than a pooled average.

## Why this is worth building rather than waiting

Upstream recognizes the general problem — [vllm-project/vllm#16969](https://github.com/vllm-project/vllm/issues/16969)
and [#29406](https://github.com/vllm-project/vllm/issues/29406) both describe large
requests monopolizing the budget and head-of-line-blocking others. Neither has produced
a shipped fix. The one shipped mitigation, `long_prefill_token_threshold`, targets the
many-small-behind-one-large shape and measurably makes this shape *worse*.

Every alternative path — different serving framework, multi-GPU architecture,
client-side admission control — was evaluated and rejected for this hardware and model
combination. See [alternatives-considered.md](alternatives-considered.md).
