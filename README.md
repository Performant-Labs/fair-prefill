# fair-prefill

A pluggable scheduler for vLLM V1 that fair-shares the per-step token budget
across concurrently **prefilling** requests, instead of letting one large
prefill monopolize it.

> **Status: design / not yet implemented.** See the
> [epic](https://github.com/Performant-Labs/fair-prefill/issues/1) for the plan.

## Documentation

| Document | What it covers |
|---|---|
| [docs/motivation.md](docs/motivation.md) | The workload, the scheduler mechanism that starves it, and the entrainment finding that makes collision *cost* — not frequency — the thing to fix |
| [docs/alternatives-considered.md](docs/alternatives-considered.md) | Every other approach evaluated and why it was rejected: config knobs, multi-GPU architectures, alternative serving frameworks, client-side admission control |
| [docs/measurement.md](docs/measurement.md) | How a change is judged to have worked, and the seven rules derived from prior false-positive results |

## The problem

vLLM V1's scheduler walks its running queue in FCFS order each step, allocating
from a single shared per-step token budget (`max_num_batched_tokens`). Nothing
caps how much of that budget one already-running request's prefill may claim, and
the queue is never reordered or round-robined. When two clients each send very
large prompts, one request's prefill can consume the entire budget for several
consecutive steps — the other client is not merely deprioritized, it is not
scheduled at all until the first finishes.

This is a poor fit for a small number of peers each carrying a large context
(for example, long-running agentic coding sessions), as opposed to the
many-small-requests-behind-one-large-request shape that existing mitigations
target.

### Why the existing knobs don't cover it

| Mechanism | Why it doesn't solve this |
|---|---|
| `long_prefill_token_threshold` | Caps a single request's per-step consumption but never changes ordering, so the same request still goes first every step — it just needs more steps to finish. Measured **worse** for two comparable-size peers. |
| Prefix caching | Only helps when prompts share reusable prefixes. Two independently-evolving conversations share almost nothing. |
| `--scheduling-policy priority` | Reorders the *waiting* queue and selects preemption victims, but force-preemption only fires on KV-cache block exhaustion. With KV cache far from full, it never triggers. It also never reorders the running queue's own per-step walk. |
| Raising `max_num_batched_tokens` | Genuinely helps — it shortens how long any one prefill monopolizes the budget, lowering collision *probability*. But it does not change what happens during a collision, and is bounded by KV-cache memory. |

Upstream tracks this general problem in
[vllm-project/vllm#16969](https://github.com/vllm-project/vllm/issues/16969) and
[#29406](https://github.com/vllm-project/vllm/issues/29406); no fix has shipped.

## The approach

vLLM V1 supports swapping in a custom scheduler class via `--scheduler-cls`
(added in [vllm-project/vllm#14466](https://github.com/vllm-project/vllm/pull/14466)),
so this ships as a plugin — **no fork of vLLM required**.

The scheduler subclasses `AsyncScheduler` and overrides `schedule()` to divide
the per-step token budget fairly among requests that are actively prefilling,
while leaving decode-phase requests their normal small allocation.

### Known constraints

- `SchedulerInterface` is explicitly **not a public API**. vLLM logs a warning
  when a custom scheduler class is configured, and compatibility across versions
  is not guaranteed. This project pins to a specific vLLM version and treats
  every upgrade as requiring a re-audit.
- A custom scheduler must subclass **`AsyncScheduler`**, not the base
  `Scheduler` — vLLM's own warning states that subclassing `Scheduler` disables
  async scheduling and degrades performance.
- Overriding `schedule()` means carrying a copy of a large, stateful method.
  That copy freezes upstream's logic at a point in time and must be diffed
  against upstream on every version bump.

## Prior art

- **FairBatching** ([arXiv 2510.14392](https://arxiv.org/abs/2510.14392)) — research
  on a V1-compatible pluggable scheduler addressing prefill/decode fairness with
  adaptive capacity. Not a drop-in equal-budget splitter and not a maintained
  package, but the closest known prior work; worth reading before writing new logic.
- **CacheAffinityScheduler** ([RFC vllm-project/vllm#42185](https://github.com/vllm-project/vllm/issues/42185)) —
  a different pluggable scheduler that reorders the *waiting* queue by prefix-cache
  affinity. Solves a different problem, but demonstrates `--scheduler-cls` in real use.

No existing implementation of per-step fair-share across concurrently prefilling
requests was found.

## License

Apache-2.0
