# Research Snapshot: PPO-Lite Baseline

Date: 2026-05-15

## Summary

Today's build moves the repository from a pure scaffold/diagnostic harness toward
a first learning-baseline artifact. The custom gridworld now has a small
PPO-lite actor-critic attacker baseline trained against the environment's greedy
defender. This baseline is intentionally separate from the EBM strategy loop so
that future comparisons can distinguish ordinary policy learning from latent
strategy generation.

The result is operational but not yet strong: PPO-lite trains, logs finite
losses, and appears in the baseline table, but it is still caught in every eval
episode under the current short debug run.

## What Changed

- Added a PPO-lite actor-critic baseline for `custom_gridworld_v0`.
- Added clipped PPO loss, value loss, entropy regularization, GAE-style returns,
  advantage normalization, minibatch updates, and deterministic evaluation.
- Added `examples/train_ppo_baseline.py` and completed
  `configs/gridworld_ppo_baseline.yaml`.
- Extended the baseline comparison table with `ppo_baseline`.
- Added unit and smoke tests for PPO shapes, losses, return/advantage
  computation, training execution, artifact writing, and comparison schema.

Generated PPO artifacts are written locally to:

- `outputs/public/ppo_baseline/metrics.json`
- `outputs/public/baselines/metrics.json`

These generated files remain ignored by Git by default.

## Current PPO-Lite Result

Configuration:

- Environment: 10x10 custom attacker-defender gridworld.
- Defender: built-in greedy pursuit defender.
- PPO total steps: 512.
- Rollout steps per update: 128.
- Updates: 4.
- Evaluation episodes: 5.

PPO-lite metrics:

| Metric | Value |
| --- | ---: |
| Episode return | -1.180 |
| Win rate | 0.000 |
| Goal rate | 0.000 |
| Catch rate | 1.000 |
| Timeout rate | 0.000 |
| Policy loss | -0.0037 |
| Value loss | 0.0819 |
| Entropy | 1.5972 |
| Training episodes completed | 25 |
| Updates | 4 |

The losses are finite and the training loop is live. The policy is not yet
competitive.

## Baseline Comparison

Current public baseline table from `examples/compare_baselines.py`:

| Baseline | Episode Return | Win Rate | Goal Rate | Catch Rate | Timeout Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_policy` | -1.208 | 0.000 | 0.000 | 1.000 | 0.000 |
| `direct_goal_heuristic` | 0.910 | 1.000 | 1.000 | 0.000 | 0.000 |
| `day2_strategy_loop` | -1.220 | 0.000 | 0.000 | 1.000 | 0.000 |
| `ppo_baseline` | -1.180 | 0.000 | 0.000 | 1.000 | 0.000 |

Diagnostic interpretation: PPO-lite is slightly less bad than the random and
Day 2 strategy-loop rows in this one short run, but it still fails the task. The
direct-goal heuristic remains the strongest baseline by a wide margin.

## Why This Matters

This is a useful build-day step because the repo now has:

- a standard model-free learning baseline,
- comparable public metrics,
- tests around the learning objective,
- a runnable example that writes artifacts,
- and a clean separation between policy learning and EBM strategy generation.

That makes future claims harder to fool ourselves with: the strategy loop must
eventually beat not only random and heuristics, but also a tuned learning
baseline.

## Limitations

This snapshot does not claim:

- PPO is tuned,
- PPO beats the heuristic,
- the EBM strategy loop improves policy learning,
- robust multi-agent reasoning,
- or true Nash exploitability.

The current PPO run is short, single-config, and diagnostic. It is best read as
"the baseline is alive and measurable."

## Immediate Next Steps

1. Run PPO-lite across multiple seeds and report mean/std metrics.
2. Increase total PPO steps and inspect whether goal-reaching emerges.
3. Add held-out grid layouts so direct-goal overfitting is visible.
4. Compare strategy-conditioned PPO against unconditioned PPO.
5. Feed stronger policy rollouts back into the strategy buffer and evaluator.
