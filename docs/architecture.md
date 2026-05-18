# Architecture

This project separates high-level strategy generation from low-level action execution. The first implementation is a small, inspectable scaffold rather than a production RL system.

## Energy-Based Strategy Generator

The EBM defines a scalar energy function over latent strategy embeddings. Low-energy embeddings are intended to represent promising high-level strategies. Candidate strategies are sampled with Langevin dynamics, and named heuristic embeddings are included for debugging.

Current implementation:

- `strategy_games.models.ebm.EnergyMLP`
- `strategy_games.strategies.sampler.langevin_sample`
- `strategy_games.strategies.embeddings.named_strategy_embedding`

## Game-Theoretic Evaluator

The evaluator scores candidate attacker strategies against a sampled set of defender responses. It computes average-case value, worst-case value, robustness, and an `exploitability_proxy`. This is not exact Nash exploitability.

Current implementation:

- `strategy_games.evaluation.best_response.GameTheoreticEvaluator`
- Heuristic defender responses: aggressive, evasive, feint, patient, direct_goal

## Strategy-Conditioned Policy

The policy receives an encoded environment state and a latent strategy embedding, then outputs action logits. The Day 2 loop applies a minimal REINFORCE-style update from collected rollout log-probabilities. This is useful for proving the training path is live, but it is not a replacement for a careful PPO implementation.

Current implementation:

- `strategy_games.models.policy.StrategyConditionedPolicy`
- `strategy_games.models.policy.RandomPolicy`

## World Model

The world model is a clean interface for learned transition and reward prediction. The first evaluator still uses true environment rollouts. The Day 2 training loop fits the world model on observed one-step transitions, leaving imaginary rollout evaluation as a future extension.

Current implementation:

- `strategy_games.models.world_model.LearnedWorldModel`

## Strategy Buffer

The buffer stores strategy embeddings and metadata: return, robustness score, exploitability proxy, iteration, timestamp, label, and optional extra metadata. It supports sampling positive strategies for future EBM training.

Current implementation:

- `strategy_games.strategies.buffer.StrategyBuffer`
- `strategy_games.strategies.buffer.StrategyRecord`

## Experiment Runner

The experiment runner loads YAML configs and executes a small Generate -> Evaluate -> Execute -> Update loop. It records selection metrics, rollout summaries, candidate diversity, and update losses for the policy, EBM, and world model. When config logging is enabled, it writes a public run directory containing `iterations.jsonl`, `metrics.json`, and `config.yaml`.

Current implementation:

- `strategy_games.experiments.runner.run_from_config`
- `strategy_games.training.train_loop.run_training_loop`

## Observability and Evaluation Harness

The Day 3-7 harness adds public tooling around the training loop:

- Experiment logging for reproducible run artifacts
- Rollout traces and matplotlib grid path plots
- Baseline comparison across random, direct-goal, and strategy-loop methods
- Payoff matrices for named strategy-vs-opponent heuristic evaluation
- Benchmark registry and adapters for custom gridworld and optional PettingZoo Pursuit runs

Benchmark adapters expose a comparable result schema while preserving environment-specific semantics. The custom gridworld remains the primary research environment; PettingZoo Pursuit is a secondary transfer benchmark behind an optional dependency.

## Pursuit Trace Viewer

The multi-evader pursuit/evasion demo is trace-first: a custom scripted rollout produces a validated `PursuitTrace`, and the CLI exporter, fixture, tests, docs, and optional Streamlit viewer consume that artifact. The viewer is intended for inspecting environment dynamics, scripted policy behavior, and trace-level metrics. It does not demonstrate learned robustness, optimality, or exact game-theoretic guarantees.

Current implementation:

- `strategy_games.traces.pursuit_trace.PursuitTrace`
- `strategy_games.envs.multi_evader_pursuit.MultiEvaderPursuitEnv`
- `strategy_games.rollouts.pursuit_runner.run_scripted_pursuit_rollout`
