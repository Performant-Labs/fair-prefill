# SPDX-License-Identifier: Apache-2.0
"""fair-prefill: a pluggable vLLM V1 scheduler for fair per-step budget sharing.

Importing this package does **not** import vLLM. That is deliberate: it lets
packaging, metadata, and the qualname constant be tested in environments
without a vLLM install (CI, a dev laptop). The scheduler itself lives in
``fair_prefill.scheduler`` and does require vLLM.
"""

__version__ = "0.0.1"

#: The value to pass to vLLM's ``--scheduler-cls``.
#:
#: vLLM resolves this string with ``resolve_obj_by_qualname()``, so it is
#: **user-facing configuration**, not an internal detail. Renaming the module or
#: the class breaks every deployment that references it: treat any change here
#: as a breaking change and call it out in release notes.
SCHEDULER_QUALNAME = "fair_prefill.scheduler.FairPrefillScheduler"

__all__ = ["SCHEDULER_QUALNAME", "__version__"]
