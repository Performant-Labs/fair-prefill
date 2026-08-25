# Alternatives considered

Everything evaluated before deciding to write a custom scheduler, and why each was
rejected. Recorded so these are not silently re-litigated later.

Findings marked **measured** were tested against the real workload. Findings marked
**source-verified** were confirmed by reading the installed vLLM source or official
documentation. Findings marked **researched** rest on third-party documentation and
were not independently reproduced.

---

## 1. vLLM configuration knobs

### `long_prefill_token_threshold` — rejected, measured worse

The community-standard answer to "a long prefill is starving other requests"
([vllm-project/vllm#15419](https://github.com/vllm-project/vllm/pull/15419)). Caps how
many tokens a request classified as "long" may consume per step, even while running.

**Result: substantially worse than baseline.** Contention indicators improved — the flag
does exactly what it advertises — but wall-clock TTFT got significantly worse.

**Why.** The cap limits a request's per-step *maximum*; it does not change *ordering*.
The same request is still first in the running queue every step, so it still goes first —
it just now needs several times more steps to finish its own prefill, and every one of
those steps still interleaves with the other peer. Two peers of comparable size end up
trading small slices back and forth instead of one finishing and yielding.

This is the clearest instance of the shape mismatch: with many small requests behind one
large one, capping the large one lets the small ones slip through, which is a real win.
With two large peers, there are no small requests to slip through — capping just
lengthens the contention window.

**Important reframing — the mechanism may be right, the value was wrong.** A static
threshold applies even when a request is running **alone**, so an uncontended prefill was
stretched across several times more steps than necessary for no benefit at all. "Full
budget when uncontended, split when contended" is exactly what a fixed value cannot
express.

That observation is the basis of [issue #14](https://github.com/Performant-Labs/fair-prefill/issues/14):
if the threshold is set *dynamically per step* — to roughly the per-step budget divided
by the number of actively prefilling requests — it may deliver fair-share directly, with
no need to override `schedule()` at all. That would remove this project's largest
maintenance liability. The spike will confirm or kill it; if the mechanism itself turns
out to be the problem rather than the static value, that result gets recorded here.

**Warning for the design in #9:** an overly aggressive split can still reproduce this
failure. Splitting only helps while the shares stay large enough that per-step overhead
does not dominate, which is why the allocation policy needs a floor.

### Prefix caching — rejected, negligible effect

Expected to help, since a growing conversation resends mostly-unchanged history.

**Result: single-digit-percent cache hit rate.** Effectively no benefit.

**Why.** Prefix caching requires a shared *literal prefix*. Two independent agentic
sessions diverge almost immediately — different tool outputs, different file contents,
different timestamps — so there is very little reusable prefix either within or across
sessions. The mechanism suits shared system prompts or templated RAG traffic across many
short-lived requests, not two long uniquely-evolving conversations.

Left enabled (it is the default and does no harm), but it is not a fix.

### `--scheduling-policy priority` — rejected, source-verified inapplicable

Initially the most promising remaining lever: it genuinely sorts both queues by
`(priority, arrival_time)` and can force-preempt a lower-priority running request. The
plan was to have clients alternate priority each turn to force fair turn-taking.

**Rejected after reading the scheduler source, before implementing.** Two findings:

1. **Force-preemption only fires on KV-cache block exhaustion.** The preemption branch is
   reached only when block allocation returns nothing. With KV utilization moderate, that
   condition effectively never occurs on this workload, so no preemption ever happens
   regardless of priority values.
2. **Priority never reorders the running queue's own per-step walk.** Requests are
   appended to the running list in admission order and iterated in that order. Priority
   affects which *waiting* request is admitted next and which request is chosen as a
   preemption victim — neither of which is the bottleneck here.

Worth recording as a near-miss: the documentation-level description ("priority reorders
the queues, and can preempt") reads as though it would solve this. Only the source shows
that neither mechanism engages for this workload.

### Raising `max_num_batched_tokens` — adopted, partial and non-durable

The one config change that measurably helped, and it remains in the deployment. See
[motivation.md](motivation.md) for the full result and the decay finding: it reduces
collision *frequency*, not collision *cost*, and frequency drifts back up over
multi-hour sessions via cadence entrainment.

Also bounded — the activation memory for larger batches competes with KV cache, and
past a point the engine cannot allocate a full-length KV cache at startup at all. On a
memory-constrained single GPU that ceiling arrives quickly.

**This is the baseline fair-prefill must beat, not a solved problem.**

---

## 2. Multi-GPU architectural approaches — not available

### Disaggregated prefill/decode — requires ≥2 GPUs

Running prefill and decode on separate engine instances so a large prefill cannot block
decoding. Every shipped vLLM implementation (KV connectors and the associated transfer
layers) assumes separate GPUs or nodes with a KV-transfer fabric between them.

Single-GPU time-slicing between prefill and decode phases exists in the research
literature but is not merged into vLLM and is not installable.

**Not applicable to a single-GPU deployment.**

### Decode context parallelism — requires ≥2 GPUs

Splits the KV cache across GPUs along the sequence dimension to avoid duplication under
long-context concurrency. Benchmarked on large multi-GPU nodes. **Not applicable.**

---

## 3. Alternative serving frameworks — all rejected for this stack

The binding constraints are: **Intel Arc / XPU backend**, **GPTQ-Int4 quantization**, and
**MTP-style speculative decoding**. A framework that drops any of these is not a
substitute, because each is load-bearing for the deployment's throughput.

### TensorRT-LLM — hard blocker

CUDA/NVIDIA only. There is no path on an Intel Arc GPU. Excluded categorically.

### lmdeploy — no evidence of support

No documentation and no user reports of Intel XPU/Arc support were found. Treated as
unsupported rather than merely undocumented.

### SGLang — attractive scheduler, unusable backend for this stack

The most genuinely tempting alternative: SGLang's cache-aware, prefill-first scheduling
is architecturally better suited to this problem than vLLM's FCFS walk, and third-party
comparisons show it degrading less under concurrent load.

**But its Intel XPU backend cannot currently run this stack.** From SGLang's own XPU
documentation, **speculative decoding is listed as "Not yet implemented"** on that
backend. GPTQ-Int4 on XPU and full attention/decode support appear as open roadmap items
rather than shipped, validated features, XPU kernels live in a separate package (a sign
of bolted-on rather than native support), and XPU-specific crash bugs were open
recently.

Switching would mean giving up speculative decoding **and** the quantization scheme
simultaneously, on a backend with known instability on this hardware family — to gain a
scheduler improvement. Net negative.

*Reassess if SGLang's XPU backend later ships speculative decoding and validated
GPTQ-Int4. That would genuinely change this analysis.*

### intel/llm-scaler (`llm-scaler-vllm`) — same problem, possibly better base

Intel's own downstream vLLM build for Arc Pro hardware, with Intel-tuned kernels, GPTQ
support, and some model-specific speculative decoding. Being downstream vLLM, **it
inherits the identical V1 scheduler behavior** and does not address fairness.

Not a competitor to this project — a possible *foundation* for it. Worth revisiting as a
base image if Intel kernel performance justifies it; the scheduler plugin would layer on
top unchanged.

### IPEX-LLM — insufficient evidence for this combination

Continuous batching via a vLLM path with low-bit/GPTQ support. Weaker evidence for
robust MTP-equivalent speculative decoding under concurrent long-context load on this
hardware. Not a clear improvement, and shares the same underlying scheduler limitation.

### OpenVINO Model Server / GenAI — production-grade, no advantage here

Supports continuous batching, paged attention, chunked prefill, INT4, and speculative
decoding on Intel GPUs, and is production-oriented. But nothing indicates it solves
per-step budget fairness for a few large concurrent contexts any better than a tuned
vLLM, and migrating carries its own substantial cost.

**Framework verdict: no alternative offers per-step fairness *and* feature parity for
XPU + GPTQ-Int4 + speculative decoding. Staying on vLLM and fixing the scheduler is the
only path that preserves the whole stack.**

---

## 4. Client-side approaches — real but insufficient

### Blind periodic delay / jitter — rejected, simulated worse

Idea: add randomized delay before each large request, as a symmetry-breaker against the
entrainment described in [motivation.md](motivation.md) — the standard trick for
desynchronizing networked peers.

**Simulated against the real empirical timing distributions: worse at every jitter
magnitude tested.** Collision frequency does fall as jitter grows, but total wall-clock
time to complete the same work rises monotonically. Delay is paid on *every* request,
including the substantial fraction that were never going to collide, and that cost
exceeds the savings.

### Targeted admission control ("wait if the peer is busy") — viable, small

Idea: hold a request client-side only when the other peer is actually mid-request,
releasing it when the GPU frees, so it experiences uncontended processing time instead
of paying the collision penalty.

**Simulated at a low-single-digit-percent net wall-clock improvement.** Real and positive
— the collision penalty genuinely exceeds the wait — but modest, and it requires building
and maintaining an admission-control proxy in front of the server. It also cannot help
the case it most needs to: it serializes the peers rather than letting them genuinely
share the GPU.

**Not rejected on principle** — it composes with fair-share scheduling and could be
revisited later. It is simply not a substitute for fixing the scheduler.

### KV cache offload (e.g. LMCache) — untried, plausible, different axis

Offload an idle peer's KV cache to host memory and restore it when its turn returns,
avoiding recompute. A third-party benchmark on very different (large multi-GPU) hardware
reported meaningful TTFT gains for concurrent long-context agentic sessions, and
single-GPU host-memory offload does not require the storage fabric that benchmark used.

**Never tested on this hardware.** Unknown whether it transfers to a single Arc GPU with
fp8 KV cache and speculative decoding. Attacks a genuinely different cost (recompute on
turn resumption) than fair-share (contention during overlap), so the two are
complementary rather than competing.

Recorded as the most promising *untried* lever if fair-share scheduling proves
insufficient.

---

## Summary

| Approach | Status | Reason |
|---|---|---|
| `long_prefill_token_threshold` | Rejected | Measured worse; wrong workload shape |
| Prefix caching | Kept, ineffective | Negligible hit rate; no shared prefixes |
| `--scheduling-policy priority` | Rejected | Source-verified: never engages here |
| Raise `max_num_batched_tokens` | **Adopted, partial** | Helps, but decays via entrainment; memory-bounded |
| Disaggregated prefill/decode | Unavailable | Requires ≥2 GPUs |
| Decode context parallelism | Unavailable | Requires ≥2 GPUs |
| TensorRT-LLM | Excluded | CUDA-only |
| lmdeploy | Excluded | No XPU support found |
| SGLang | Rejected (revisit) | XPU backend lacks speculative decoding + validated GPTQ-Int4 |
| intel/llm-scaler | Not a fix | Downstream vLLM; same scheduler. Possible base image |
| IPEX-LLM | Rejected | Insufficient evidence for this combination |
| OpenVINO MS/GenAI | Rejected | No fairness advantage; migration cost |
| Blind client-side jitter | Rejected | Simulated worse at every magnitude |
| Targeted admission control | Deferred | ~low single digit %; composes with this project |
| KV cache offload | Untried | Different axis; complementary; most promising untried lever |
| **Custom `--scheduler-cls`** | **Chosen** | Only path preserving XPU + GPTQ-Int4 + speculative decoding |
