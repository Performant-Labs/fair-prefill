# fair-prefill

A pluggable scheduler for vLLM V1 that fair-shares the per-step token budget
across concurrently **prefilling** requests, instead of letting one large
prefill monopolize it.

> **Status: design / not yet implemented.** See the
> [epic](https://github.com/Performant-Labs/fair-prefill/issues/1) for the plan.

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
