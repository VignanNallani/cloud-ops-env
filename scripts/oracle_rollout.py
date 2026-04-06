"""
Oracle rollout for the CloudOpsEnvironment.

Purpose:
- Verify the environment can be solved to 1.0 for easy/medium/hard.
- Provide a deterministic, auditable action sequence and per-step progress table.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Any

from cloud_ops_env.env import CloudOpsAction, CloudOpsEnvironment, SecurityStatus, TIER_SPECS


TIERS: tuple[str, ...] = ("nano", "standard", "performance")


def _infer_tier_from_cost_perf(hourly_cost_usd: float, performance_units: float) -> str | None:
    """Infer current tier from (cost, performance) pair with small tolerance."""
    for tier, (c, p) in TIER_SPECS.items():
        if math.isclose(hourly_cost_usd, c, rel_tol=0.0, abs_tol=1e-9) and math.isclose(
            performance_units, p, rel_tol=0.0, abs_tol=1e-9
        ):
            return tier
    return None


def _fleet_ratio_for_assignment(active_servers: list[Any], assignment: dict[str, str]) -> float:
    """Compute fleet cost/perf ratio deterministically for a tier assignment."""
    total_cost = 0.0
    total_perf = 0.0
    for s in active_servers:
        tier = assignment[s.id]
        cost, perf = TIER_SPECS[tier]
        total_cost += float(cost)
        total_perf += float(perf)
    if total_perf <= 1e-12:
        return 0.0
    return total_cost / total_perf


def _solve_hard_exact_or_best(
    active_servers: list[Any],
    target_ratio: float,
) -> tuple[dict[str, str], float, bool]:
    """
    Try to find a tier assignment that matches target_ratio exactly.
    If none exists, choose the assignment that minimizes relative error.
    Returns: (assignment_by_server_id, best_rel_err, is_exact)
    """
    best_assignment: dict[str, str] = {}
    best_rel_err = float("inf")
    found_exact = False

    # Brute force: up to 3 active servers, 3 tiers each -> 27 combinations.
    server_ids = [s.id for s in active_servers]
    tier_product = itertools.product(TIERS, repeat=len(server_ids))
    for tiers_choice in tier_product:
        assignment = {sid: tier for sid, tier in zip(server_ids, tiers_choice)}
        ratio = _fleet_ratio_for_assignment(active_servers, assignment)
        if target_ratio == 0.0:
            continue
        rel_err = abs(ratio - target_ratio) / target_ratio
        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_assignment = assignment
        if ratio == target_ratio:
            found_exact = True
            best_assignment = assignment
            best_rel_err = 0.0
            break

    return best_assignment, best_rel_err, found_exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CloudOpsEnvironment()
    obs = env.reset(seed=args.seed)

    servers = obs.servers
    idle_servers = [s for s in servers if s.cpu_utilization_percent < 5.0 and s.active]
    vulnerable_servers = [s for s in servers if s.security_status == SecurityStatus.SSH_EXPOSED_WORLD and s.active]

    # Hard task is about the aggregate cost/perf ratio over active servers.
    # Since oracle will terminate idle servers first, compute "misaligned" among the post-easy active set.
    post_easy_active = [s for s in servers if s.active and s.id not in {x.id for x in idle_servers}]
    target_ratio = float(obs.target_cost_performance_ratio)

    chosen_assignment, best_rel_err, is_exact = _solve_hard_exact_or_best(
        active_servers=post_easy_active,
        target_ratio=target_ratio,
    )

    misaligned_servers = [
        s
        for s in post_easy_active
        if chosen_assignment.get(s.id) is not None
        and _infer_tier_from_cost_perf(s.hourly_cost_usd, s.performance_units) != chosen_assignment[s.id]
    ]

    print("Initial identification")
    print(f"Idle (easy): {[s.id for s in idle_servers]}")
    print(f"Vulnerable (medium): {[s.id for s in vulnerable_servers]}")
    print(
        f"Misaligned (hard): {[s.id for s in misaligned_servers]} "
        f"(target_ratio={target_ratio:.12f}, rel_err_best={best_rel_err:.6g}, exact_match={is_exact})"
    )
    print()

    rows: list[dict[str, Any]] = []
    step_no = 0

    def _record_step(action: CloudOpsAction, reward: float | None, next_obs: Any) -> None:
        nonlocal step_no
        step_no += 1
        scores = next_obs.grader_scores
        total_progress = (scores["easy"] + scores["medium"] + scores["hard"]) / 3.0
        rows.append(
            {
                "step": step_no,
                "action": action.model_dump(mode="json"),
                "reward": float(reward) if reward is not None else None,
                "total_progress": float(total_progress),
                "scores": {k: float(v) for k, v in scores.items()},
            }
        )

    # 4.1 Solve Easy: terminate all idle servers.
    for s in idle_servers:
        action = CloudOpsAction(command="terminate_server", server_id=s.id, instance_tier="standard")
        result = env.step(action)
        _record_step(action, result.reward, result)
        obs = result

    # 4.2 Solve Medium: fix all vulnerable servers.
    # (After easy, some vulnerable servers might have been terminated; skip those.)
    current_servers = {s.id: s for s in obs.servers if s.active}
    for s in vulnerable_servers:
        if s.id not in current_servers:
            continue
        action = CloudOpsAction(command="fix_ssh_exposure", server_id=s.id, instance_tier="standard")
        result = env.step(action)
        _record_step(action, result.reward, result)
        obs = result

    # 4.3 Solve Hard: set instance tiers to match target ratio.
    # Execute assignment on the currently-active post-easy servers (and only those).
    # chosen_assignment was computed over the post-easy active set.
    current_servers = {s.id: s for s in obs.servers if s.active}
    for sid, desired_tier in chosen_assignment.items():
        if sid not in current_servers:
            continue
        current_s = current_servers[sid]
        inferred = _infer_tier_from_cost_perf(current_s.hourly_cost_usd, current_s.performance_units)
        if inferred == desired_tier:
            continue
        action = CloudOpsAction(
            command="set_instance_tier",
            server_id=sid,
            instance_tier=desired_tier,
        )
        result = env.step(action)
        _record_step(action, result.reward, result)
        obs = result

    # 5. Print step-by-step table.
    print("Oracle rollout steps")
    header = ["Step #", "Action", "Reward", "Total Progress"]
    print(f"{header[0]:<8} | {header[1]:<55} | {header[2]:<10} | {header[3]:<14}")
    print("-" * 110)
    for r in rows:
        action_str = f"{r['action']['command']}:{r['action'].get('server_id','')}:{r['action'].get('instance_tier','')}"
        reward_str = "None" if r["reward"] is None else f"{r['reward']:.6f}"
        print(
            f"{r['step']:<8} | {action_str:<55} | {reward_str:<10} | {r['total_progress']:<14.6f}"
        )

    final_scores = obs.grader_scores
    print()
    print("Final grader scores:", {k: float(v) for k, v in final_scores.items()})

    # 6. Assertions.
    assert float(final_scores["easy"]) == 1.0, f"Easy not solved: {final_scores['easy']}"
    assert float(final_scores["medium"]) == 1.0, f"Medium not solved: {final_scores['medium']}"
    assert float(final_scores["hard"]) == 1.0, f"Hard not solved: {final_scores['hard']}"


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        # Specific analysis request: explain mathematical reason hard cannot hit exactly 1.0.
        print()
        print("Solvability check failed.")
        print(str(e))
        print()
        print("env.py analysis (mathematical fix suggestion):")
        print(
            "- The Hard grader computes rel_err = abs(current_ratio - target_ratio) / target_ratio and returns "
            "1.0 only when rel_err == 0.0, i.e., when current_ratio equals target_ratio exactly."
        )
        print(
            "- But CloudOpsEnvironment generates target_ratio as base_ratio * jitter where jitter is a random float in "
            "[0.78, 1.22]. The agent can only set ratios using discrete tier pairs, so exact equality is not "
            "guaranteed."
        )
        print()
        print("Proposed fixes (choose one):")
        print(
            "A) Make the target always reachable: in _spawn_episode, set target_cost_performance_ratio to be computed "
            "from a randomly chosen *reachable tier assignment* (exactly using the same tier cost/perf numbers), "
            "instead of base_ratio * uniform jitter."
        )
        print(
            "B) Add tolerance in the grader: in HardCostPerformanceGrader.score, return 1.0 when rel_err <= epsilon "
            "(e.g., epsilon=0.005), rather than requiring rel_err == 0.0."
        )
        raise

