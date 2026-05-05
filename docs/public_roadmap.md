# Public Roadmap

This repository is currently a research scaffold for early experiments in energy-based strategy generation and game-theoretic evaluation.

## Current

- Configurable attacker-defender gridworld
- Named heuristic strategy embeddings
- Energy MLP over strategy embeddings
- Langevin sampler
- Approximate sampled-response evaluator
- Strategy buffer
- Debug training loop with lightweight policy, EBM, and world-model updates
- Public experiment logging
- Rollout trace and path visualization
- Baseline comparison harness
- Named-strategy payoff matrix
- Unit and smoke tests

## Near Term

- Replace the lightweight policy update with a tested PPO baseline
- Improve EBM training from buffer positives and sampler negatives
- Add richer rollout logging and visualization summaries
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
