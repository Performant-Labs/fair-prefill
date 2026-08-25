# fair-prefill

An investigation into whether fair-sharing vLLM V1's per-step token budget
across concurrently prefilling requests can help when two clients each send very
large prompts.

> ## Status: not pursued — the premise was tested and refuted
>
> **Fair-sharing the per-step token budget cannot help the request that finishes
> last.** Total prefill work is conserved, so the last finisher is bound by total
> work rather than by scheduling order. Sharing only delays the *first* finisher.
>
> Measured, two equal prompts: stock finishes at steps 7 and 13; fair-share at 13
> and 13. The starved request gains nothing. This held at every arrival stagger
> tested, and with three peers it is worse — stock `[7, 13, 19]` versus
> fair-share `[19, 19, 19]`.
>
> Four separate arguments for the approach were each tested and refuted:
>
> | Argument | Result |
> |---|---|
> | Equal-split helps the starved request | No — last finisher unchanged |
> | SRPT would do better | No — identical to stock in every scenario |
> | It cuts tail latency | No — **worst case is invariant** across all policies; variance falls only because good outcomes get worse |
> | It protects a concurrent decode client | No — decode served in 20/20 steps under both; it is never starved |
>
> Production data agrees: the median collision penalty is **2.10× per input
> token**, matching the ~2× that work conservation predicts for the second of two
> peers.
>
> ### Then the root cause turned out to be something else entirely
>
> Follow-up measurement (2026-08-25) showed the bottleneck was never the per-step
> budget at all. It is **KV cache residency**.
>
> The cache holds ~139k tokens. Two agentic sessions each carrying ~100k of
> context need ~200k, so each turn evicts the other session's cached prefix and
> the next turn recomputes its entire prompt at ~539 tok/s. Across 195 contended
> production requests: **4%** prefix-cache hit, and 78,788 tokens ÷ 539 tok/s =
> 146s against 153.9s observed. Uncontended requests finish a *larger* 106k-token
> prompt in 6.2s, which at that throughput is only possible as a cache hit.
>
> So the "collision penalty" was cache hit versus full recompute, not contention.
> Raising `max_num_batched_tokens` is also not the lever — it was tried
> (8192 → 12288 → 16384) with no demonstrable TTFT improvement.
>
> **The real lever is keeping prefixes resident**: KV offload to host RAM, a
> smaller `max_model_len`, or less per-session context.
>
> ### The lesson worth keeping
>
> The harness reproduced production's symptom exactly — same `running=1,
> waiting=1` signature, same starvation pattern. That was convincing, and it was
> measuring the wrong thing: it had no KV cache and no eviction, so it could not
> have shown the real cause. **Reproducing a symptom is not reproducing its
> mechanism.**
>
> The code here is a working `--scheduler-cls` plugin and a GPU-free harness that
> drives vLLM's real scheduler. Both are kept because the harness is reusable and
> the negative result is worth being able to reproduce. See
> [#1](https://github.com/Performant-Labs/fair-prefill/issues/1) for the full
> test series.

## Motivation

vLLM V1's scheduler walks its running queue in FCFS order each step, drawing from one
shared per-step token budget (`max_num_batched_tokens`). Nothing caps how much of that
budget an already-running request's prefill may claim. When two clients each send very
large prompts, one prefill can consume the entire budget for several consecutive steps —
the other client isn't deprioritized, it simply isn't scheduled at all until the first
finishes.

Existing mitigations target the *many small requests behind one large one* shape and
misfire when every participant is large.

Full background in [`docs/`](docs/):

- **[motivation.md](docs/motivation.md)** — the workload, the mechanism, and why the
  durable problem is collision *cost* rather than collision *frequency*
- **[alternatives-considered.md](docs/alternatives-considered.md)** — every other
  approach evaluated and why it was rejected
- **[prior-art.md](docs/prior-art.md)** — existing `--scheduler-cls` plugins and
  fairness-scheduling research
- **[measurement.md](docs/measurement.md)** — how a change is judged to have worked

## Approach

vLLM V1 supports swapping in a custom scheduler class via `--scheduler-cls`
(added in [vllm-project/vllm#14466](https://github.com/vllm-project/vllm/pull/14466)),
so this ships as a plugin — **no fork of vLLM required**.

The scheduler subclasses `AsyncScheduler` and overrides `schedule()` to divide the
per-step token budget fairly among requests that are actively prefilling, while leaving
decode-phase requests their normal small allocation.

## Loading it into vLLM

Pass the class by qualname:

```
--scheduler-cls fair_prefill.scheduler.FairPrefillScheduler
```

vLLM resolves that string with `resolve_obj_by_qualname()`, so **the package must be
importable by the serving process** — the flag alone does nothing if the import fails.
That string is user-facing configuration; renaming the module or class is a breaking
change. `make qualname` prints the current value.

Confirm vLLM actually resolved *your* class rather than silently falling back: on startup
it logs a warning naming the configured scheduler. No warning means no custom scheduler.

### Getting the package into the container

**Development** — mount the source and put it on the path, so edits need only a restart,
not an image rebuild:

```bash
docker run ... -v /path/to/fair-prefill:/fair-prefill:ro -e PYTHONPATH=/fair-prefill ...
```

**Release** — install into the image (`pip install fair-prefill`). Deferred to
[#8](https://github.com/Performant-Labs/fair-prefill/issues/8).

## Development

```bash
make check           # lint + tests, what CI runs
make test            # pytest; vLLM-dependent tests skip when vLLM is absent
make test-container  # full suite inside the serving image, vLLM present
```

`import fair_prefill` deliberately does **not** import vLLM, so packaging and metadata
are testable without it. Only `fair_prefill.scheduler` requires vLLM; those tests are
marked `requires_vllm` and skip elsewhere — the skip is reported in the run header, since
a suite that silently tests nothing is worse than a failing one.

## Constraints

- `SchedulerInterface` is explicitly **not a public API**; vLLM logs a warning when a
  custom scheduler is configured and does not guarantee cross-version compatibility.
  This project pins to a specific vLLM version and treats every upgrade as a re-audit.
- A custom scheduler must subclass **`AsyncScheduler`**, not the base `Scheduler` —
  vLLM's own warning states that subclassing `Scheduler` disables async scheduling and
  degrades performance. Note that `async_scheduling=None` means *auto-decide*, not
  *off*: vLLM enables it unless an incompatibility applies, so many deployments are
  already running `AsyncScheduler` without ever passing the flag. Resolve the actual
  engine config to find out rather than inferring from the command line.
- Overriding `schedule()` means carrying a copy of a large, stateful method that must be
  diffed against upstream on every version bump.

## License

Apache-2.0
