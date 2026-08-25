# Measurement methodology

How this project decides whether a change worked. Written before the work started, so
the criteria are not chosen after seeing results.

The short version: **three prior mitigations for this problem produced results that
looked good and were not.** Each failure mode below is something that actually happened,
and each rule exists to catch it.

## Rule 1 — measure real traffic, not benchmarks

Synthetic load generators do not reproduce the behavior that matters here. The dominant
effect — cadence entrainment between two agentic clients — is a property of how real
clients *respond* to latency: they do work between calls, and the duration of that work
depends on what the model returned. A synthetic driver with fixed or randomly-drawn
inter-arrival times cannot produce it, and will therefore report a fix as working when
it does not durably work.

Fast deterministic tests (see the scheduler harness issue) exist for iteration speed and
correctness. They are **not** evidence a change improved anything. Only real concurrent
traffic settles that.

## Rule 2 — samples must span multiple hours

**This is the rule that matters most, and the one violated to greatest cost.**

A previous change measured over roughly its first hour showed an enormous improvement in
median TTFT. Over the following hours it substantially decayed as collision frequency
rose to a new, higher equilibrium. The one-hour number was not wrong — it was accurately
describing a transient state that the system then left.

Any sign-off based on a sub-hour sample is rejected regardless of how good it looks.

## Rule 3 — report distributions, and track them over time

Report median, mean, and tail **separately**. They can and do move in opposite
directions: in one measurement the median improved dramatically while the mean stayed
high, because a shrinking minority of requests were still paying the full original cost.
A single summary statistic hid the fact that the problem was only partly addressed.

Also plot the key metrics **over the measurement window**, not just aggregated across it.
A metric that drifts steadily across four hours and a metric that is stable at the same
average are completely different outcomes, and averaging erases the distinction.

## Rule 4 — the headline metric is TTFT conditioned on collision

Partition requests by whether another large request was already in flight when this one
arrived:

- **No-collision requests** — measures whether the change broke the easy case.
- **Collision requests** — measures whether the change fixed the actual problem.

Report these **separately**, plus the collision rate itself.

**Why this partition is the whole point.** Raising `max_num_batched_tokens` improved the
pooled median enormously while leaving the cost of an actual collision essentially
untouched — it worked by making collisions *rarer*, not cheaper. Since collision
frequency then drifted back up on its own, the pooled median flattered a change whose
durable benefit was much smaller.

Fair-share scheduling targets collision **cost**. So:

> **TTFT-given-collision is the primary result. If it does not improve, the project did
> not work — regardless of what the pooled median says.**

Collision rate should be reported too, but as context. It is driven substantially by
client behavior, not only by the scheduler.

## Rule 5 — guard metrics must not regress

Fairness is a redistribution, and redistribution can cost aggregate efficiency. Every
comparison reports:

- **Decode throughput** — must be unaffected; the scheduler change should touch prefill
  allocation only.
- **Speculative decoding acceptance rate** — the canary for scheduling bugs corrupting
  draft-token bookkeeping. A fair-share split landing a prefill chunk boundary in the
  wrong place could silently degrade acceptance without any crash or error. Establish the
  stock value during backbone-parity measurement so there is something to compare to.
- **Total system throughput** — if fairness costs aggregate throughput, that is a
  legitimate tradeoff, but it must be stated with numbers, never omitted.

## Rule 6 — verify against installed source, not documentation

During the investigation that produced this project, two independent research passes each
recommended configuration flags that **did not exist** in the installed version — one set
was stale from a previous major version, another was described in a way that implied
behavior the source did not implement.

Before relying on any flag, class, or interface: read it in the vLLM actually installed
in the serving image. Documentation, blog posts, and research summaries are leads, not
evidence. The `--scheduling-policy priority` rejection in
[alternatives-considered.md](alternatives-considered.md) is a direct product of this rule
— it looked correct in the docs and was ruled out only by reading the code.

**Absence of a signal is not a signal.** A related failure: this project was initially
planned around the belief that the deployment ran the base `Scheduler`, because
`--async-scheduling` was not passed and no log line mentioned it. Both observations were
worthless. `async_scheduling=None` means *auto-decide*, and vLLM's resolution logs a
warning on every path that **disables** it while logging nothing when it **enables** it —
so silence in the logs is exactly what an enabled deployment looks like. Resolving the
real engine config showed async scheduling had been on the whole time.

When a check comes back empty, establish that the check would have produced output had
the thing been true, before concluding anything from the emptiness.

## Rule 7 — report negative results plainly

Three mitigations were tested and rejected on measurements before this project started.
A fourth rejection is a completely legitimate outcome and far more valuable than a
result that only holds for the first hour.

If fair-share scheduling does not improve TTFT-given-collision, say so, record why, and
close it out. Do not quietly reframe the criteria to make it a success, and correct
earlier optimistic conclusions rather than letting them stand.

## What gets captured per run

- Per-request: arrival time, TTFT, total latency, input tokens, output tokens, finish
  reason, and which client issued it.
- Server-side time series: requests running, requests waiting (by reason), KV cache
  utilization, speculative decoding acceptance rate.
- Configuration: exact vLLM version, all scheduler-relevant flags, plugin version and
  commit.

Raw per-request data is archived alongside every summary. Summary statistics without the
underlying data cannot be re-partitioned later — and the collision partition in Rule 4
was devised *after* the run that first needed it, which was only possible because the
raw data still existed.
