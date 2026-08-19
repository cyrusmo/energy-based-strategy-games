# Research Snapshot: Hardened PPO, Aligned Execution, Held-Out Layouts

Date: 2026-08-18

Status: current diagnostic snapshot. Numeric tables below are generated from
`outputs/public/` artifacts produced in this cycle. Refresh those artifacts
before citing.

## Summary

This cycle closed the three near-term research-loop gaps: PPO was a live but
non-learning baseline, the strategy loop could not execute selected strategies,
and the default map hid direct-goal overfitting.

After potential-based reward shaping and a 4096-step budget, PPO reaches the
goal on the default 10x10 map. After aligning evaluator and executor action
sources, the strategy loop matches the direct-goal heuristic on that same map.
Held-out layouts show that both still fail when the defender starts on the
direct path. Medium-term items (world-model rollouts, PSRO, richer evaluation)
stay deferred.

## What Changed

- PPO: goal-distance potential shaping, `reward_scale`, higher entropy, 4096-step
  default budget, periodic eval curves, and a public training diagnosis.
- Strategy loop: `executor_action_source: aligned` so named strategies are
  executed with the same heuristic map used to score them; stronger behavior
  cloning and shaping in `configs/gridworld_day3.yaml`.
- Held-out layouts: `default`, `open_right`, `intercept_path`,
  `goal_near_defender`, `cramped`, with a public comparison protocol.
- Reports: learning-study and held-out artifacts regenerated; this snapshot
  replaces the May 15 PPO-lite diagnostic as the current readout.

## PPO Result

Configuration: 10x10 default map, greedy defender, 4096 steps, shaping_coef 1.0,
entropy_coef 0.05, 8 eval episodes.

| Metric | Value |
| --- | ---: |
| Episode return | 0.910 |
| Win / goal rate | 1.000 |
| Catch rate | 0.000 |
| Diagnosis | `learning_path_live` |

Goal-reaching appears around 1024 steps in the update history. This is a
default-map result, not a held-out claim.

## Learning Study (3 seeds)

From `outputs/public/learning_study/`:

| Method | Mean return | Success (goal_rate) | Failure mode |
| --- | ---: | ---: | --- |
| `ppo_baseline` | 0.91 | 1.0 | `learning_path_live` |
| `strategy_loop` | 0.91 | 1.0 | `learning_path_live` |

The strategy loop still selects `aggressive` on every seed. That label shares
the direct-to-goal action map, so default-map success is execution alignment,
not a diverse latent strategy space. Buffer diversity remains 0.0.

## Held-Out Layouts

From `outputs/public/heldout_layouts/`:

| Baseline | Default goal rate | Held-out goal rate | Drop |
| --- | ---: | ---: | ---: |
| `direct_goal_heuristic` | 1.00 | 0.75 | 0.25 |
| `strategy_loop` | 1.00 | 0.75 | 0.25 |
| `random_policy` | 0.00 | 0.00 | 0.00 |

`direct_goal_overfitting_visible` is true. The only layout that breaks
direct-goal is `intercept_path` (defender on the goal column). The strategy
loop fails there too, because aligned heuristic execution copies the same
path. `goal_near_defender` selected `evasive` once; other layouts still
selected `aggressive`.

## Baseline Table (default map)

From `outputs/public/baselines/metrics.json`:

| Baseline | Episode return | Win rate | Catch rate |
| --- | ---: | ---: | ---: |
| `random_policy` | -1.208 | 0.000 | 1.000 |
| `direct_goal_heuristic` | 0.910 | 1.000 | 0.000 |
| `day2_strategy_loop` | 0.910 | 1.000 | 0.000 |
| `ppo_baseline` | 0.910 | 1.000 | 0.000 |

## Interpretation

The harness can now show learning and competent strategy execution on the
default geometry. It can also show that this success is not transfer: the
same methods lose when the greedy defender sits on the obvious path.

This does **not** mean the EBM discovered a robust strategy, that PPO is
tuned, or that `exploitability_proxy` is Nash exploitability.

## Decision Gate

A learning method does not yet beat direct-goal on held-out maps. It ties the
heuristic, including on the layout the heuristic fails.

Keep iterating on execution variety and layout-sensitive selection. Do not
open world-model imagined rollouts, PSRO populations, or richer domains until
a method wins on `intercept_path` without collapsing to a single named
heuristic.

## Immediate Next Steps

1. Make named heuristic action maps behaviorally distinct (`aggressive` vs
   `direct_goal`).
2. Prefer layout-sensitive selection so `evasive` can win on `intercept_path`.
3. Evaluate the hardened PPO on held-out layouts.
4. Only then reconsider world-model rollouts or PSRO-style comparisons.
