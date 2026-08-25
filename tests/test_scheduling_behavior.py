# SPDX-License-Identifier: Apache-2.0
"""Scheduler behavior under concurrent prefills.

These tests encode measured behavior of vLLM's real scheduler, not aspirations.
Several of them assert that fair-sharing does **not** help -- see
`test_fair_share_cannot_help_equal_peers`, which is the finding that reshaped
this project. Do not "fix" a failure here by loosening the assertion without
first re-deriving the arithmetic in its docstring.

Run inside the serving image: `make test-container`.
"""

import pytest

pytestmark = pytest.mark.requires_vllm


def _harness():
    import harness

    return harness


def _fair_share_hook(h, budget):
    """The candidate policy: split the budget across pending prefills."""

    def hook(sched):
        n = h.pending_prefills(sched)
        sched.scheduler_config.long_prefill_token_threshold = (
            budget // n if n > 1 else 0
        )

    return hook


# --------------------------------------------------------------------------
# The problem this project exists to address
# --------------------------------------------------------------------------


def test_stock_starves_the_second_large_prefill():
    """One large prefill takes the entire per-step budget for several
    consecutive steps; a second concurrent request gets literally zero.

    This is the production symptom reproduced in miniature. It is a property of
    the scheduler's greedy in-order walk over the running queue.
    """
    h = _harness()
    sched = h.build_scheduler()
    reqs = h.make_requests(2)
    for r in reqs:
        sched.add_request(r)
    a, b = (r.request_id for r in reqs)

    timeline = h.drive(sched, 25)

    # The first several steps go entirely to one request.
    starved_steps = 0
    for step in timeline:
        if step.get(a, 0) == h.BUDGET and step.get(b, 0) == 0:
            starved_steps += 1
        else:
            break
    assert starved_steps >= 4, (
        f"expected sustained monopolization, saw {starved_steps} steps: {timeline[:8]}"
    )


def test_fair_share_makes_both_prefills_progress():
    """The mechanism works: a dynamic threshold splits the budget.

    Note what this does and does not establish. It shows the *knob functions* --
    both requests receive tokens in the same step instead of one taking
    everything. It says nothing about whether that is desirable, which is what
    the next two tests examine.
    """
    h = _harness()
    sched = h.build_scheduler()
    reqs = h.make_requests(2)
    for r in reqs:
        sched.add_request(r)
    ids = [r.request_id for r in reqs]

    timeline = h.drive(sched, 25, before_step=_fair_share_hook(h, h.BUDGET))

    both = h.steps_with_all_progressing(timeline, ids)
    assert both >= 10, f"expected sustained interleaving, got {both} steps"
    first = timeline[0]
    assert first[ids[0]] == first[ids[1]] == h.BUDGET // 2


# --------------------------------------------------------------------------
# Why that is not enough
# --------------------------------------------------------------------------


def test_fair_share_cannot_help_equal_peers():
    """With two comparably-sized prefills, fair-sharing cannot improve the
    later request, and makes the earlier one worse.

    Total prefill work is conserved. Two 1600-token prompts at 256 tokens/step
    is ~13 steps of work however it is divided, so whichever request finishes
    *last* is bound by total work, not by ordering. Sharing can only delay the
    first finisher.

    Measured: stock finishes at steps 7 and 13 (mean 10). Fair-share finishes at
    13 and 13 (mean 13). The starved request gains nothing; the other loses six
    steps.

    This is the central negative result for this project's original premise, and
    it holds at every arrival stagger tested.
    """
    h = _harness()
    prompt = h.LARGE_PROMPT

    def run(hook):
        sched = h.build_scheduler()
        reqs = h.make_requests(2)
        for r in reqs:
            sched.add_request(r)
        ids = [r.request_id for r in reqs]
        tl = h.drive(sched, 40, before_step=hook)
        return [h.steps_until_prefill_done(tl, i, prompt) for i in ids]

    stock = run(None)
    fair = run(_fair_share_hook(h, h.BUDGET))

    # The request that finishes last is unchanged -- work-bound, not order-bound.
    assert max(fair) >= max(stock), (
        f"fair-share unexpectedly improved the last finisher: {stock} -> {fair}"
    )
    # And the first finisher is strictly worse off.
    assert min(fair) > min(stock), (
        f"expected the early finisher to regress: {stock} -> {fair}"
    )
    # Mean completion is worse.
    assert sum(fair) / len(fair) > sum(stock) / len(stock)


@pytest.mark.parametrize(
    "big,small,fair_should_win",
    [
        (1600, 256, True),  # 6:1  -- large win for the small request
        (1600, 512, True),  # 3:1  -- clear win
        (1600, 800, False),  # 2:1  -- a wash
        (1200, 1200, False),  # 1:1  -- fair-share loses
    ],
)
def test_fair_share_pays_off_only_with_size_asymmetry(big, small, fair_should_win):
    """Fair-share's benefit tracks the size ratio between concurrent prefills.

    Measured mean completion (stock vs fair): 6:1 -> 7.5 vs 5.0; 3:1 -> 8.0 vs
    6.5; 2:1 -> 8.5 vs 8.5; 1:1 -> 7.5 vs 10.0.

    So the policy helps exactly the small-request-behind-large-request shape
    that vLLM's built-in `long_prefill_token_threshold` already targets, and
    hurts the two-comparable-peers shape this project was created for.
    """
    h = _harness()
    from tests.v1.core.utils import create_requests

    def run(hook):
        sched = h.build_scheduler()
        a = create_requests(
            1, num_tokens=big, block_size=16, max_tokens=8, req_ids=["BIG"]
        )[0]
        b = create_requests(
            1, num_tokens=small, block_size=16, max_tokens=8, req_ids=["small"]
        )[0]
        sched.add_request(a)
        sched.add_request(b)
        tl = h.drive(sched, 40, before_step=hook)
        return (
            h.steps_until_prefill_done(tl, "BIG", big),
            h.steps_until_prefill_done(tl, "small", small),
        )

    s_big, s_small = run(None)
    f_big, f_small = run(_fair_share_hook(h, h.BUDGET))
    stock_mean = (s_big + s_small) / 2
    fair_mean = (f_big + f_small) / 2

    if fair_should_win:
        assert fair_mean < stock_mean, (
            f"{big}:{small} expected fair-share to win, "
            f"stock={stock_mean} fair={fair_mean}"
        )
    else:
        assert fair_mean >= stock_mean, (
            f"{big}:{small} expected fair-share not to win, "
            f"stock={stock_mean} fair={fair_mean}"
        )


# --------------------------------------------------------------------------
# Policy comparison: is any reordering better?
# --------------------------------------------------------------------------


def _srpt_hook(h):
    """Shortest-remaining-processing-time: serve the request closest to done.

    Implemented by reordering the running list, since the greedy walk consumes
    the budget in list order. SRPT is classically optimal for mean flow time,
    which makes it the strongest realistic alternative to equal-split.
    """

    def hook(sched):
        def remaining(r):
            left = r.num_prompt_tokens - r.num_computed_tokens
            return left if left > 0 else 10**9

        sched.running.sort(key=remaining)
        sched.scheduler_config.long_prefill_token_threshold = 0

    return hook


def _completion_steps(h, sizes, hook, stagger=0):
    from tests.v1.core.utils import create_requests

    sched = h.build_scheduler()
    reqs = [
        create_requests(
            1, num_tokens=n, block_size=16, max_tokens=8, req_ids=[f"r{i}"]
        )[0]
        for i, n in enumerate(sizes)
    ]
    sched.add_request(reqs[0])
    timeline = h.drive(sched, stagger, before_step=hook) if stagger else []
    for r in reqs[1:]:
        sched.add_request(r)
    timeline += h.drive(sched, 60, before_step=hook)
    return [
        h.steps_until_prefill_done(timeline, f"r{i}", n) for i, n in enumerate(sizes)
    ]


def test_srpt_is_indistinguishable_from_fcfs():
    """SRPT buys nothing over stock in any scenario tested.

    With equal-sized requests SRPT degenerates to FCFS (all remaining times are
    equal, so the ordering never changes). In the asymmetric case vLLM's arrival
    order already happens to be favourable, so it matches there too.

    Measured, as [r0, r1] completion steps: equal peers 1600/1600 -> [7, 13] for
    both FCFS and SRPT; staggered -> [7, 13] both; asymmetric 1600/400 ->
    [7, 8] both; three peers -> [7, 13, 19] both.

    This closes off "a smarter reordering would fix it": the classically optimal
    policy for mean flow time is already what the stock scheduler does here.
    """
    h = _harness()
    for sizes, stagger in (
        ((1600, 1600), 0),
        ((1600, 1600), 3),
        ((1600, 400), 0),
        ((1600, 1600, 1600), 0),
    ):
        fcfs = _completion_steps(h, sizes, None, stagger)
        srpt = _completion_steps(h, sizes, _srpt_hook(h), stagger)
        assert fcfs == srpt, f"sizes={sizes} stagger={stagger}: {fcfs} vs {srpt}"


def test_worst_case_completion_is_policy_invariant():
    """No policy improves the WORST completion step -- only the best ones.

    This is the sharpest form of the negative result, and it matters because
    "fair-share reduces tail latency" is the most appealing remaining argument
    for the approach. It does reduce variance, but entirely by making good
    outcomes worse, never by making the bad outcome better.

    Measured worst-case completion step:

        equal peers   FCFS 13   equal-split 13   SRPT 13
        staggered     FCFS 13   equal-split 13   SRPT 13
        three peers   FCFS 19   equal-split 19   SRPT 19

    Three peers is the starkest: FCFS yields [7, 13, 19], equal-split [19, 19,
    19]. Mean degrades 13 -> 19 while the worst case does not move at all. Fair
    sharing converted "one turn in three is fast" into "every turn is the
    slowest case".
    """
    h = _harness()
    for sizes, stagger in (
        ((1600, 1600), 0),
        ((1600, 1600), 3),
        ((1600, 1600, 1600), 0),
    ):
        worst = {
            name: max(_completion_steps(h, sizes, hook, stagger))
            for name, hook in (
                ("fcfs", None),
                ("equal", _fair_share_hook(h, h.BUDGET)),
                ("srpt", _srpt_hook(h)),
            )
        }
        assert len(set(worst.values())) == 1, (
            f"sizes={sizes} stagger={stagger}: worst case differs across "
            f"policies: {worst}"
        )


# --------------------------------------------------------------------------
# Guards on the harness itself
# --------------------------------------------------------------------------


def test_solo_prefill_is_untouched_by_the_policy():
    """With one request there is nothing to share, so the policy must be inert."""
    h = _harness()

    def run(hook):
        sched = h.build_scheduler()
        req = h.make_requests(1)[0]
        sched.add_request(req)
        return h.drive(sched, 20, before_step=hook)

    assert run(None) == run(_fair_share_hook(h, h.BUDGET))


def test_prompt_over_context_limit_is_rejected_loudly():
    """A prompt longer than the model context silently stalls the scheduler
    rather than erroring, so the harness refuses it up front."""
    h = _harness()
    with pytest.raises(ValueError, match="context"):
        h.make_requests(1, prompt_tokens=h.CONTEXT_LIMIT + 1)
