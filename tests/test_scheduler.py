# SPDX-License-Identifier: Apache-2.0
"""Scheduler tests. Require a real vLLM; run inside the serving image.

Run them with ``make test-container``, or directly::

    docker exec -e PYTHONPATH=/fair-prefill <container> \\
        python3 -m pytest /fair-prefill/tests

These cover what can be asserted about the class itself. Proving vLLM actually
*loads* it is issue #4, and proving it schedules identically to stock is the
harness in issue #6.
"""

import pytest

pytestmark = pytest.mark.requires_vllm


def test_resolves_through_vllms_own_loader():
    """Resolve the qualname the same way vLLM does, not with a plain import.

    vLLM calls ``resolve_obj_by_qualname()`` on the ``--scheduler-cls`` string.
    Using that function here means the test exercises the real lookup path, so
    a qualname that imports fine but that vLLM cannot resolve still fails.
    """
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from fair_prefill import SCHEDULER_QUALNAME

    cls = resolve_obj_by_qualname(SCHEDULER_QUALNAME)
    assert cls.__name__ == "FairPrefillScheduler"


def test_subclasses_async_scheduler_not_scheduler():
    """Subclassing ``Scheduler`` instead of ``AsyncScheduler`` silently disables
    async scheduling. Async scheduling is on by default wherever no
    incompatibility applies, so this regression would cost real performance
    while everything still appeared to work."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from fair_prefill.scheduler import FairPrefillScheduler

    assert issubclass(FairPrefillScheduler, AsyncScheduler)


def test_satisfies_the_scheduler_interface():
    from vllm.v1.core.sched.interface import SchedulerInterface

    from fair_prefill.scheduler import FairPrefillScheduler

    assert issubclass(FairPrefillScheduler, SchedulerInterface)


def test_is_still_a_pure_passthrough():
    """Milestone 1 requires behavioral identity with stock.

    Fails the moment someone overrides a method, which is the intended prompt
    to go re-read #14 before modifying ``schedule()`` -- the spike may make the
    override unnecessary entirely.
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from fair_prefill.scheduler import FairPrefillScheduler

    overrides = {
        name
        for name, attr in vars(FairPrefillScheduler).items()
        if not name.startswith("__") and callable(attr)
    }
    assert not overrides, (
        f"FairPrefillScheduler overrides {sorted(overrides)}; milestone 1 is "
        f"pass-through only. Base class is {AsyncScheduler.__name__}."
    )
