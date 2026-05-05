# Experiments

The initial experiments are designed to validate the research decomposition, not to claim final performance.

## Domain

The first domain is a configurable attacker-defender gridworld. The attacker starts at one grid location, the defender starts elsewhere, and the attacker receives reward for reaching a goal before being intercepted.

Default setup:

- Grid: 10x10
- Attacker start: `(0, 0)`
- Defender start: `(9, 9)`
- Goal: `(9, 0)`
- Terminal conditions: goal reached, defender catch, or max steps

## Baselines

Initial baselines:

- Uniform random attacker policy
- Direct-to-goal heuristic attacker
- Gaussian latent strategy sampler
- PPO baseline scaffold
- Lightweight strategy-conditioned policy-gradient update

Planned baselines:

- PPO without strategy conditioning
- Strategy-conditioned PPO with random latent strategies
- PSRO-style population baseline
- Heuristic defender/opponent populations with held-out responses

## Metrics

Core metrics:

- `episode_return`
- `win_rate`
- `goal_rate`
- `catch_rate`
- `average_case_value`
- `worst_case_value`
- `robustness_score`
- `exploitability_proxy`
- `strategy_diversity`

Planned metrics:

- Zero-shot transfer across maps and goals
- Performance against held-out opponent policies
- Payoff matrix coverage
- Embedding clustering quality
- Rollout behavioral diversity

## Ablations

Near-term ablations:

- EBM sampling vs Gaussian latent sampling
- With vs without strategy buffer positive sampling
- With vs without robustness-aware selection
- Number of candidate strategies per iteration
- Number and type of sampled opponent responses
- With vs without world-model fitting
- Lightweight REINFORCE update vs PPO

## Reporting Standard

Public reports should state whether metrics are exact or approximate. The current `exploitability_proxy` should not be presented as true Nash exploitability.

## Public Artifacts

Config-driven runs can write public artifacts under `outputs/public/`:

- `iterations.jsonl`: one JSON object per training iteration
- `metrics.json`: aggregate run metrics and final update losses
- `config.yaml`: saved config snapshot
- `trace.txt`: line-oriented rollout trace for public demos
- `trajectory.png`: simple grid path visualization
- `matrix.json`: named strategy-vs-opponent payoff matrix

Generated demo artifacts are ignored by default so curated results can be promoted deliberately.
