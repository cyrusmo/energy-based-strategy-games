# Research Snapshot: Energy-Based Strategy Games

Date: 2026-05-11

## Summary

This repository is an early-stage research scaffold for separating latent strategy
generation from low-level policy execution in adversarial multi-agent games. The
current implementation includes a custom attacker-defender gridworld, an EBM-based
strategy sampler, sampled-response evaluation, rollout visualization, baseline
comparison, payoff matrices, and benchmark logging.

The current diagnostic result is intentionally modest: the harness is working, and
it surfaces that the simple direct-goal heuristic currently outperforms the early
strategy loop. This is useful signal, not a performance claim.

## Motivation

The central research question is whether high-level strategies can be generated,
evaluated, selected, and executed as a distinct layer above low-level action
policies. In this scaffold, an Energy-Based Model proposes latent strategy
embeddings, an evaluator scores those candidates against sampled opponents, and a
strategy-conditioned policy executes the selected strategy in the environment.

This decomposition is meant to make multi-agent reasoning easier to inspect:
strategy generation, game-theoretic evaluation, execution quality, and learning
updates can be debugged separately before making stronger claims.

## Architecture

The current loop is:

1. Generate candidate strategy embeddings with an EBM and Langevin sampler.
2. Evaluate candidates against sampled opponent heuristics.
3. Select the strongest candidate by robustness-aware score.
4. Execute a strategy-conditioned policy in the gridworld.
5. Store rollout summaries and strategy metadata.
6. Apply lightweight update hooks for policy, EBM, and world model components.

Implemented public-facing components include:

- `strategy_games.envs.gridworld`: custom attacker-defender pursuit/evasion game.
- `strategy_games.models.ebm`: MLP energy function over strategy embeddings.
- `strategy_games.strategies.sampler`: random and Langevin strategy sampling.
- `strategy_games.evaluation`: sampled best-response, robustness, and
  exploitability proxy metrics.
- `strategy_games.training.train_loop`: runnable generate-evaluate-execute-update
  scaffold.
- `strategy_games.experiments`: logging, metrics, baselines, payoff matrices, and
  benchmark runners.

## Environment

The primary research/debug environment is `custom_gridworld_v0`, a configurable
10x10 attacker-defender game. The attacker starts near one corner and tries to
reach a goal, while the defender tries to intercept. Episodes terminate when:

- the attacker reaches the goal,
- the defender catches the attacker,
- or the maximum step budget is reached.

This environment is intentionally simple enough to debug while still supporting
distinct strategy labels such as `aggressive`, `evasive`, `feint`, `patient`, and
`direct_goal`.

PettingZoo `pursuit_v4` is available as an optional transfer benchmark through
`pip install -e ".[bench]"`. It is secondary to the custom gridworld and should
be interpreted as a smoke-tested external adapter, not a performance result.

## Benchmark Protocol

The refreshed public diagnostics were generated with:

- `configs/benchmarks/custom_gridworld.yaml`
- `configs/benchmarks/debug_suite.yaml`
- `configs/gridworld_day3.yaml`
- `examples/visualize_rollout.py`
- `examples/compare_baselines.py`
- `examples/compute_payoff_matrix.py`

The benchmark runner writes local artifacts under `outputs/public/`, including
`results.jsonl`, `summary.json`, rollout traces, plots, baseline tables, payoff
matrices, and training-loop logs. These generated files are ignored by Git by
default so that only curated outputs are committed deliberately.

## Current Results

Custom gridworld benchmark, 3 seeds:

| Baseline | Runs | Return Mean +/- Std | Win Rate | Goal Rate | Catch Rate | Timeout Rate | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_goal_heuristic` | 3 | 0.910 +/- 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 9.00 |
| `random_policy` | 3 | -1.217 +/- 0.031 | 0.000 | 0.000 | 1.000 | 0.000 | 21.67 |
| `strategy_loop` | 3 | -1.133 +/- 0.099 | 0.000 | 0.000 | 0.889 | 0.111 | 22.33 |

Additional strategy-loop diagnostics from the same benchmark:

| Metric | Value |
| --- | ---: |
| Average-case value | 0.510 |
| Worst-case value | -1.090 |
| Exploitability proxy | 1.600 |
| Strategy diversity | 0.000 |

Training-loop artifact from `configs/gridworld_day3.yaml`:

| Metric | Value |
| --- | ---: |
| Iterations | 3 |
| Final selected label | `aggressive` |
| Buffer size | 3 |
| Buffer diversity | 0.000 |
| Mean episode return | -1.243 |
| Mean win rate | 0.000 |
| Mean goal rate | 0.000 |
| Mean catch rate | 1.000 |

The update losses are finite, which is the main requirement at this stage. The
current policy update is intentionally lightweight and should not be interpreted
as a competitive RL training result.

Optional PettingZoo Pursuit smoke benchmark, 2 seeds:

| Baseline | Runs | Return Mean +/- Std | Capture Proxy | Timeout Rate | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `strategy_loop` | 2 | -2.418 +/- 0.018 | 0.000 | 1.000 | 25.00 |
| `random_policy` | 2 | -2.438 +/- 0.013 | 0.000 | 1.000 | 25.00 |
| `direct_goal_heuristic` | 2 | -2.465 +/- 0.018 | 0.000 | 1.000 | 25.00 |

These PettingZoo rows only validate that the external benchmark adapter and
artifact schema run locally. The toy config ends by timeout for all rows, so it
should not be used as evidence of transfer performance.

## Rollout Visualization

The refreshed direct-goal rollout artifacts are:

- `outputs/public/rollout_demo/trace.txt`
- `outputs/public/rollout_demo/trajectory.png`

The text trace shows the attacker moving from `(0, 0)` to the goal at `(9, 0)` in
9 steps, with outcome `goal` and total return `0.910`.

## Payoff Matrix Summary

The named-strategy payoff matrix compares attacker strategy labels against sampled
opponent heuristic labels:

| Strategy | Aggressive Opponent | Evasive Opponent | Feint Opponent | Patient Opponent | Direct-Goal Opponent | Best Response |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `aggressive` | 0.910 | 0.910 | 0.910 | 0.910 | -1.090 | `direct_goal` |
| `evasive` | 0.910 | 0.910 | 0.910 | 0.910 | -1.090 | `direct_goal` |
| `feint` | 0.910 | 0.910 | 0.910 | 0.910 | -1.090 | `direct_goal` |
| `patient` | 0.820 | 0.820 | 0.750 | -1.260 | -1.320 | `direct_goal` |
| `direct_goal` | 0.910 | 0.910 | 0.910 | 0.910 | -1.090 | `direct_goal` |

In this diagnostic setup, the direct-goal opponent is the strongest sampled
response for every listed attacker strategy. This is a sampled-response result,
not an exact Nash exploitability calculation.

## Interpretation

The current strategy loop is live but weak. The direct-goal heuristic is stronger
in the current grid layout and reward structure, while the learned
strategy-conditioned policy is not yet competitive.

Likely reasons:

- The policy starts effectively untrained.
- The current update is a minimal REINFORCE-style hook rather than a full PPO
  implementation.
- The evaluator uses heuristic rollouts that may not yet align with the executing
  policy's learned behavior.
- The seed counts are intentionally tiny for smoke testing.
- The current sampled strategy diversity is still 0.000 in these runs.

The value of the current repo is therefore the research decomposition,
instrumentation, and benchmark readiness, not demonstrated superiority over
baselines.

## Limitations

This snapshot does not claim:

- state-of-the-art performance,
- robust multi-agent reasoning,
- true Nash exploitability,
- convergence guarantees,
- or that EBM-generated strategies outperform simple baselines.

All exploitability metrics should be read as `exploitability_proxy` values unless
an exact equilibrium computation is added later.

## Immediate Next Steps

1. Implement a real PPO baseline and a stronger strategy-conditioned policy
   update.
2. Improve evaluator/executor alignment so selected latent strategies map to
   reliable behavior.
3. Add held-out grid layouts and larger seed sweeps.
4. Add strategy diversity pressure or explicit behavior descriptors.
5. Expand PettingZoo `pursuit_v4` beyond the current smoke test into a calibrated
   external transfer benchmark.

## Short Message To Researchers

I am building a research scaffold for separating latent strategy generation from
policy execution in adversarial multi-agent games. The current version includes a
custom attacker-defender gridworld, EBM strategy sampler, sampled-response
evaluator, baseline comparison, payoff matrices, rollout visualization, and
benchmark logging. The first diagnostic results show the infrastructure is
working and also reveal that the current lightweight strategy loop is not yet
competitive with a direct-goal heuristic, which is the next research target.
