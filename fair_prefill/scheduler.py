# SPDX-License-Identifier: Apache-2.0
"""The scheduler class vLLM loads via ``--scheduler-cls``.

Load it with::

    vllm serve ... --scheduler-cls fair_prefill.scheduler.FairPrefillScheduler

vLLM resolves that string through ``resolve_obj_by_qualname()``, so this module
must be importable by the serving process — see the README for how the package
reaches the container.

Subclassing ``AsyncScheduler`` rather than ``Scheduler`` is required, not
stylistic. vLLM logs this when a custom scheduler class is configured:

    If you have subclassed Scheduler instead of AsyncScheduler, you will see
    degraded performance due to async scheduling being disabled.

Async scheduling is enabled by default on any deployment without an
incompatibility (``async_scheduling=None`` means *auto-decide*, not *off*), so
getting this wrong silently turns off something that was already running.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class FairPrefillScheduler(AsyncScheduler):
    """Fair-shares the per-step token budget across concurrently prefilling requests.

    Currently a pure pass-through: behavior is identical to the stock scheduler.
    That is intentional for milestone 1 — it establishes that the plugin loads
    and changes nothing, which is the baseline the fair-share work in milestone 2
    is measured against.

    See https://github.com/Performant-Labs/fair-prefill/issues/1
    """
