# SPDX-License-Identifier: Apache-2.0
"""Skip vLLM-dependent tests where vLLM isn't installed.

vLLM is only present inside the serving image, so CI and a dev laptop can run
the packaging tests but not the scheduler ones. Rather than let those fail
noisily, mark them ``@pytest.mark.requires_vllm``.

The skip is reported in the run summary on purpose: a silent skip is how a
suite ends up green while testing nothing.
"""

import importlib.util

import pytest

VLLM_PRESENT = importlib.util.find_spec("vllm") is not None


def pytest_collection_modifyitems(config, items):
    if VLLM_PRESENT:
        return
    skip = pytest.mark.skip(reason="vLLM not installed (run inside the serving image)")
    for item in items:
        if "requires_vllm" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config):
    where = "present" if VLLM_PRESENT else "ABSENT -- requires_vllm tests will skip"
    return f"vllm: {where}"
