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

The custom gridworld remains the primary debug and research environment. PettingZoo Pursuit can be installed as an optional transfer benchmark with `pip install -e ".[bench]"`.

## Baselines

Initial baselines:

- Uniform random attacker policy
- Direct-to-goal heuristic attacker
- Gaussian latent strategy sampler
- PPO-lite actor-critic attacker baseline
- PPO-lite `pursuer_0` baseline for multi-evader pursuit, trained against scripted evaders
- Lightweight strategy-conditioned policy-gradient update

Planned baselines:

- Hardened PPO without strategy conditioning
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
- `wall_clock_seconds`
- `survival_or_capture_rate`

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
- Lightweight REINFORCE update vs PPO-lite / hardened PPO

## Reporting Standard

Public reports should state whether metrics are exact or approximate. The current `exploitability_proxy` should not be presented as true Nash exploitability.

## Pursuit Empirical-Game Diagnostics

The multi-evader pursuit demo includes a role-specific scripted-policy comparison:

```bash
python examples/compare_pursuit_policies.py
```

After training the tiny pursuit PPO baseline, an explicit learned pursuer row can
be included:

```bash
python examples/train_ppo_pursuer.py --config configs/demo/ppo_pursuer_smoke.yaml
python examples/compare_pursuit_policies.py --include-learned-pursuer outputs/private/checkpoints/ppo_pursuer.pt
```

Rows are pursuer policies and columns are evader policies. The primary payoff is
`mean_pursuer_return`; `mean_evader_return`, capture rate, survival rate, and
average steps are reported separately. The matrix is rectangular and
general-sum, so the evaluator does not assume symmetry or zero-sum payoffs.

The empirical-game block computes row payoff against a uniform column-policy
mixture, empirical regret against that same uniform mixture, worst-case row
payoff, a maximin row policy, and a payoff-weighted row-policy ranking
distribution. This is a conservative diagnostic over scripted policies and any
explicitly provided learned policies. It is not a Nash equilibrium, CFR result,
PSRO result, learned best response, or exact exploitability estimate.

## Pursuit PPO Smoke Baseline

The first trainable multi-evader path controls only `pursuer_0` with a small
actor-critic policy. Evaders remain scripted during training. The observation
contract is `pursuit_obs/v1`, with fixed `max_pursuers=1`, `max_evaders=2`,
explicit feature order, masks, normalization, and action labels. Incompatible
agent counts are rejected instead of silently changing the model input shape.

Public outputs:

- `outputs/public/pursuit_models/ppo_pursuer/config.json`
- `outputs/public/pursuit_models/ppo_pursuer/metrics.json`

Private output:

- `outputs/private/checkpoints/ppo_pursuer.pt`

The public metrics expose `checkpoint_written`, deterministic `eval_seeds`,
training scope, observation metadata, action-space metadata, training metrics,
and fixed-seed evaluation metrics. They do not expose the private checkpoint
path by default. This baseline is intended to prove the learning/evaluation path
is live, not to claim robust pursuit performance.

## Public Artifacts

Config-driven runs can write public artifacts under `outputs/public/`:

- `iterations.jsonl`: one JSON object per training iteration
- `metrics.json`: aggregate run metrics and final update losses
- `config.yaml`: saved config snapshot
- `trace.txt`: line-oriented rollout trace for public demos
- `trace.json`: validated multi-agent pursuit/evasion trace artifact
- `trajectory.png`: simple grid path visualization
- `matrix.json`: named strategy-vs-opponent payoff matrix
- `policy_comparison.json`: pursuit scripted-policy empirical-game diagnostics
- `policy_comparison.csv`: flat public summary table for pursuit policy pairs
- `pursuit_models/ppo_pursuer/metrics.json`: first trainable pursuit PPO smoke metrics

Generated demo artifacts are ignored by default so curated results can be promoted deliberately.

## Benchmarks

Benchmark configs live under `configs/benchmarks/` and write artifacts to `outputs/public/benchmarks/`:

- `custom_gridworld.yaml`: seed sweep for the custom attacker-defender environment
- `debug_suite.yaml`: tiny benchmark used for quick local validation
- `pettingzoo_pursuit.yaml`: optional external pursuit/evasion benchmark

Benchmark rows use a shared schema across adapters: environment id, baseline, seed, return, win/goal/catch/timeout rates, survival or capture rate, steps, strategy label, sampled-response metrics when available, strategy diversity, and wall-clock time. PettingZoo rows intentionally use proxy fields where the task semantics differ from the custom attacker-goal game.
