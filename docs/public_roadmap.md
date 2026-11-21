# Public Roadmap

This repository is currently a research scaffold for early experiments in energy-based strategy generation and game-theoretic evaluation.

## Current

- Attacker-defender Gridworld v1 (kept for snapshot reproduction) and v2 (walls, two goals, held-out layouts)
- Named heuristic strategy embeddings and a parametric continuous controller
- Energy MLP over strategy embeddings with a Langevin sampler
- Strategy buffer and the generate, evaluate, execute, update training loop
- Exact Nash values, best responses and exploitability on Gridworld v2
- Gate 1 (the v2 payoff matrix is a real game) as a unit-tested property
- Gate 2 (mixed attackers on exact exploitability) with PPO, diversity-only and PSRO-lite baselines
- Energy-guided weight-space repair pilot with repair-only, position-blind and rotated-coordinate controls
- PPO baseline with potential-based shaping and training diagnosis
- Learning-study and held-out layout protocols, ablation suite, multi-seed runner
- One-step world model with open-loop imagined rollouts (not yet evaluated or used for best response)
- Pursuit/evasion trace schema, multi-evader rollout viewer and optional PettingZoo Pursuit adapter
- Public experiment logging, figures, unit and smoke tests

## Near Term

- Gate 2: make energy concentrate on a low-exploitability mixture (it is nearly flat over sampled latents)
- PPO on v2: hardened settings (`configs/gridworld_v2_ppo_hardened.yaml`) reach the goal on 11 of 15 seeds; the rest stand off against the scripted defender. Rerun Gate 2 with this baseline, and try training against a mix of defenders
- Repair pilot: train the energy without labels from the faulted layout, and plant faults in a selector with no layout-specific weights
- Measure world-model rollout error against the environment before using it for evaluation

## Medium Term

- Use the world model as a best-response engine
- Exact exploitability of draw-once-per-episode mixtures (belief-state DP)
- Self-play
- Zero-shot transfer benchmarks beyond layout swaps
- Integrate latent strategy generation into multi-evader pursuit

## Long Term

- Move beyond toy gridworlds into richer multi-agent domains
- Study how energy-based latent strategy spaces interact with equilibrium-seeking evaluation
- Compare model-free, model-based, and hybrid strategy evaluation pipelines

This roadmap is intentionally conservative. Results should be reported only after reproducible experiments are implemented and reviewed. Gate 2 currently fails, so the paper framing stays diagnostic.
