# SPDX-License-Identifier: Apache-2.0
"""Packaging invariants. These must pass without vLLM installed."""

import importlib
import importlib.util

import fair_prefill


def test_package_imports_without_vllm():
    """``import fair_prefill`` must not drag in vLLM.

    Guards a real property, not a triviality: metadata, the qualname constant,
    and (later) the version check need to be usable in environments where vLLM
    cannot be imported. If someone adds a top-level scheduler import to
    ``__init__``, this fails.
    """
    module = importlib.import_module("fair_prefill")
    assert module.__version__


def test_scheduler_qualname_matches_reality():
    """The advertised ``--scheduler-cls`` string must actually resolve.

    Checked structurally rather than by importing, so it runs without vLLM:
    the module part must exist as a file in this package, and the attribute
    name must appear in it. A rename that updates one and not the other is
    exactly the failure this catches -- and it would otherwise surface only as
    a vLLM startup crash.
    """
    module_path, _, class_name = fair_prefill.SCHEDULER_QUALNAME.rpartition(".")

    spec = importlib.util.find_spec(module_path)
    assert spec is not None, f"{module_path} is not importable"
    assert spec.origin is not None

    with open(spec.origin) as fh:
        source = fh.read()
    assert f"class {class_name}" in source, (
        f"{module_path} does not define {class_name}"
    )


def test_qualname_points_into_this_package():
    assert fair_prefill.SCHEDULER_QUALNAME.startswith("fair_prefill.")
