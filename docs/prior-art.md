# Prior art

What already exists in the `--scheduler-cls` ecosystem, and what doesn't.

**Summary: people have shipped real `--scheduler-cls` plugins, but nothing public and
maintained does the fair token-budget split this project needs.** That specific gap is
why this repo exists.

---

## Shipped plugins — proof the extension point works

These matter less for their logic than as evidence that `--scheduler-cls` is a viable,
practiced way to ship a scheduler, and as references for how to structure one.

### vllm-spyre (IBM Spyre)

One of the original drivers for making the V1 scheduler pluggable at all
([vllm-project/vllm#14466](https://github.com/vllm-project/vllm/pull/14466)). Overrides
the scheduler to accommodate hardware constraints of the Spyre accelerator.

Relevance: establishes the out-of-tree hardware-plugin pattern the extension point was
designed for. The scheduling logic itself is unrelated to fairness.

### neuralmagic/vllm-beamsearch-plugin

Ships a real `BeamSearchScheduler` loaded via `--scheduler-cls`. Production-oriented,
version-pinned, and **does not require a vLLM fork**.

Relevance: **the closest thing to a template for this project.** Same delivery model —
external package, pinned version, loaded by qualname. Worth reading before writing
packaging, versioning, or the drift-detection tooling, since it has already solved those
problems in the same context.

### CacheAffinityScheduler (RFC)

[vllm-project/vllm#42185](https://github.com/vllm-project/vllm/issues/42185) proposes a
plugin that reorders the **waiting** queue by prefix-cache affinity, so requests likely
to hit cached blocks are admitted preferentially.

Relevance: a different problem — it changes *which waiting request is admitted next*,
not how the per-step budget is divided among requests already running. It does not
address budget starvation among concurrent large prefills. Useful as a second concrete
example of plugin structure, and as evidence the community is actively extending the
scheduler rather than only forking it.

---

## Research — the fairness problem is recognized, the code isn't available

### FairBatching

[arXiv 2510.14392](https://arxiv.org/abs/2510.14392). Explicitly described as a
V1-compatible **pluggable** scheduler module (~1.6k LOC) targeting prefill/decode
fairness with adaptive budgets.

**The nearest published work to this project's goal.** It is not a drop-in equal-budget
splitter, and no public maintained GitHub package was found — only the paper. Its design
reasoning is still the best available input to the allocation-policy question (reserve
decode needs first and split the remainder, versus split evenly and reclaim), and it
should be read before that design is settled.

### Other academic schedulers

EWSJF, opportunity-cost and structure-prediction schedulers, and similar work also claim
pluggable vLLM implementations. Same pattern: papers describe the approach, no polished
public package surfaced.

### Upstream acknowledgement

[vllm-project/vllm#16969](https://github.com/vllm-project/vllm/issues/16969) and
[#29406](https://github.com/vllm-project/vllm/issues/29406) both describe large requests
monopolizing the token budget and head-of-line-blocking others. Both remain open with no
shipped fix.

The only shipped mitigation, `long_prefill_token_threshold`
([#15419](https://github.com/vllm-project/vllm/pull/15419)), is a fixed per-request cap
aimed at the many-small-behind-one-large shape — and measurably makes the
few-large-peers shape worse. See
[alternatives-considered.md](alternatives-considered.md).

---

## The gap

No maintained open-source plugin implements equal-share or round-robin allocation of the
shared `max_num_batched_tokens` budget across concurrent large-context **running**
requests.

Three distinct approaches to scheduler extension are represented above — hardware
adaptation, decoding-strategy replacement, and waiting-queue reordering. None of them
touches how the per-step budget is divided among requests already running, which is
precisely the mechanism that starves a second large prefill.

This is not a claim of novelty for its own sake. It has two practical consequences:

1. **There is no code to start from**, so the allocation logic gets written from scratch
   — with FairBatching's design reasoning as the main external input.
2. **There is no reference implementation to validate against**, so correctness rests
   entirely on this project's own harness and real-traffic measurement. That is the
   reasoning behind [measurement.md](measurement.md).

---

## Revisit triggers

Re-check this page if any of the following happens, since each would change the build
decision:

- FairBatching (or comparable research) publishes a maintained implementation.
- Upstream ships a per-step fairness mechanism against #16969 or #29406.
- A `--scheduler-cls` plugin appears that does budget-level fair-share.
