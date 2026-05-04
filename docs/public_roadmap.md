# Public Roadmap

This repository is currently a research scaffold for early experiments in energy-based strategy generation and game-theoretic evaluation.

## Current

- Configurable attacker-defender gridworld
- Named heuristic strategy embeddings
- Energy MLP over strategy embeddings
- Langevin sampler
- Approximate sampled-response evaluator
- Strategy buffer
- Debug training loop
- Unit and smoke tests

## Near Term

- Implement policy optimization for the strategy-conditioned policy
- Train the EBM from buffer positives and sampler negatives
- Add richer rollout logging and visualization
- Add held-out gridworld layouts
- Implement clean PPO baseline
- Add simple payoff-matrix experiments

## Medium Term

- Improve approximate best-response search
- Add learned world-model rollouts
- Add zero-shot transfer benchmarks
- Add PSRO-style population comparisons
- Add more interpretable strategy embedding visualizations

## Long Term

- Move beyond toy gridworlds into richer multi-agent domains
- Study how energy-based latent strategy spaces interact with equilibrium-seeking evaluation
- Compare model-free, model-based, and hybrid strategy evaluation pipelines

This roadmap is intentionally conservative. Results should be reported only after reproducible experiments are implemented and reviewed.
