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

The policy receives an encoded environment state and a latent strategy embedding, then outputs action logits. The current policy is intentionally small and untrained in the debug loop.

Current implementation:

- `strategy_games.models.policy.StrategyConditionedPolicy`
- `strategy_games.models.policy.RandomPolicy`

## World Model

The world model is a clean interface for future learned transition and reward prediction. The first evaluator uses true environment rollouts, while the world model remains a placeholder.

Current implementation:

- `strategy_games.models.world_model.LearnedWorldModel`

## Strategy Buffer

The buffer stores strategy embeddings and metadata: return, robustness score, exploitability proxy, iteration, timestamp, label, and optional extra metadata. It supports sampling positive strategies for future EBM training.

Current implementation:

- `strategy_games.strategies.buffer.StrategyBuffer`
- `strategy_games.strategies.buffer.StrategyRecord`

## Experiment Runner

The experiment runner loads YAML configs and executes a small Generate -> Evaluate -> Execute -> Update loop. It is intended to make early experiments reproducible before deeper training code is added.

Current implementation:

- `strategy_games.experiments.runner.run_from_config`
- `strategy_games.training.train_loop.run_training_loop`
