# Research Snapshot: Mixed-Strategy Gate 2

Date: 2026-10-19

Status: current diagnostic snapshot. Numeric tables below are generated from
`outputs/public/gate2/` artifacts produced in this cycle.

## Summary

Cycle 2 compared deterministic attackers on exact exploitability, a metric only
mixed strategies can win. An exact best-response defender that knows a
committed script always catches it, so every method sat near 1.6--1.8.

This cycle evaluates every method as a per-state mixed attacker (Markov
behavioral mixture). The Nash-oracle row has exploitability 0, confirming the
gap is closable. Gate 2 still fails: the EBM mixture beats diversity-only but
does not separate from softmax PPO.

## Deterministic-attacker diagnosis (Cycle 2)

From `outputs/public/gate2` before this cycle (schema `gate2/v2`):

| Method | Exploitability |
| --- | ---: |
| PPO (argmax) | 1.606 |
| Strategy loop (one latent/label) | 1.740 |
| Diversity-only | 1.797 |
| PSRO-lite | 1.811 |

Seeds 0, 2, 3, 4 selected `aggressive` at train exploitability 1.866. The Nash
attacker mix was already computed inside `_nash_table` and discarded.

## Mixed-strategy Gate 2 (schema `gate2/v3`)

Five seeds, 8 strategy iterations, PPO 4096 steps. Primary method is the EBM
Langevin mixture (K=16, weights `softmax(-E(z))`).

| Method | Kind | Exploitability | 95% CI | Value vs BR | Held-out goal | Support |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Nash oracle | reference | 0.000 | [0.000, 0.000] | 0.746 | 0.960 | 1.0 |
| EBM mixture | primary | 1.573 | [1.478, 1.668] | -1.027 | 0.128 | 15.9 |
| PPO softmax | baseline | 1.598 | [1.567, 1.630] | -1.127 | 0.000 | 1.0 |
| PPO argmax | continuity | 1.606 | [1.606, 1.606] | -1.190 | 0.000 | 1.0 |
| Uniform random | reference | 1.633 | [1.633, 1.633] | -1.069 | 0.000 | 5.0 |
| Buffer mixture | secondary | 1.726 | [1.724, 1.727] | -1.122 | 0.267 | 7.8 |
| Diversity-only | baseline | 1.762 | [1.692, 1.832] | -1.128 | 0.357 | 7.1 |
| PSRO-lite | baseline | 1.811 | [1.811, 1.811] | -1.200 | 0.200 | 1.0 |

- `passed`: false
- `framing`: diagnostic
- `beats_ppo`: false
- `beats_diversity_only`: true

The EBM mixture has almost full support (15.9 of 16 samples), so energy is
nearly flat over the Langevin cloud. It is slightly less exploitable than PPO
on the mean, but the CIs overlap. The buffer mixture stays near the
deterministic-script band.

The exploitability learning curve (`outputs/public/gate2/curves.json`) is the
buffer-mixture prefix versus cumulative env steps. Figures:
`outputs/public/paper/figures/gate2_summary.png` and
`outputs/public/paper/figures/exploitability_vs_steps.png`.

## Framing decision

Keep the paper diagnostic. The reportable result is that a zero-exploitability
attacker exists (Nash oracle) and that evaluating the EBM as a distribution
moves the method below diversity-only, but mass on low-energy latents does not
yet concentrate on a low-exploitability mixture.

Future work (not this cycle): PPO hardening on v2, belief-state draw-once
mixtures, self-play, world-model best response.
