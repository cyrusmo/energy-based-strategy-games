# Research Snapshot: Winnable Game and Honest Gate 2

Date: 2026-10-18

Status: current diagnostic snapshot. Numeric tables below are generated from
`outputs/public/` artifacts produced in this cycle. Refresh those artifacts
before citing.

## Summary

The exact oracle showed that the first Gridworld v2 layouts were unwinnable
under optimal defense: Nash value equalled the timeout return, so hiding was
equilibrium play. Gate 2 then punished that Nash hider by scoring held-out goal
rate, and its PPO/PSRO numbers were harness bugs.

This cycle made the train map winnable (`defender_period=2`), rewrote Gate 2
around exact exploitability, and re-ran five seeds with PPO at 4096 steps.
Gate 2 still fails. The honest result is that robustness-aware selection on a
now-winnable game does not beat PPO, and every method remains highly
exploitable against an exact best-response defender.

## What the artifacts said before the retune

Exact Nash on the original equal-speed layouts (about 2 s each):

| Layout | Nash value | Timeout return |
| --- | ---: | ---: |
| `corridor_split` | -0.40 | -0.40 |
| `evasive_detour` | -0.36 | -0.36 |
| `patient_hold` | -0.36 | -0.36 |

Scripted attackers had exploitability 0.68--0.80 because scoring got caught
(value versus BR about -1.1). The strategy loop selected a negative-directness
latent that hid (return -0.40, goal 0, timeout 1.0). That was Nash-optimal, not
a collapse bug.

Harness bugs in the first Gate 2 table:

- PPO and PSRO held-out goal rate 0.267 was the `direct_goal` fallback:
  `train_ppo_baseline` did not return a policy, and the PSRO mix was ignored.
- Strategy-loop `env_steps` was 0.
- `corridor_split` was flagged `held_out=True` because the protocol compared
  against the v1 train name `default`.
- Ablation `summary.json` reported `buffer_diversity=0.0` while per-run
  `metrics.json` had about 2.66; aggregation read the wrong key.

## Retune (Gate 1(d))

`defender_period=2` gives the attacker a 2:1 speed ratio. Gate 1(a)--(c) still
pass. Gate 1(d) now holds: train Nash is above timeout, and one held-out map
has negative Nash.

| Layout | Nash | Timeout | Period |
| --- | ---: | ---: | ---: |
| `corridor_split` (train) | 0.746 | -0.400 | 2 |
| `evasive_detour` | 0.029 | -0.360 | 2 |
| `feint_switch` | 0.900 | -0.380 | 2 |
| `patient_hold` | 0.880 | -0.360 | 2 |
| `open_bottom` | -0.400 | -0.400 | 1 |

Named heuristics remain highly exploitable (0.70--2.08). An exact defender
catches every committed script.

## Gate 2 (exact exploitability, 5 seeds, PPO 4096)

From `outputs/public/gate2/summary.json` (`schema_version: gate2/v2`):

| Method | Exploitability | 95% CI | Value vs BR | Held-out goal | Env. steps |
| --- | ---: | --- | ---: | ---: | ---: |
| PPO | 1.606 | [1.606, 1.606] | -1.190 | 0.333 | 4096 |
| Strategy loop | 1.740 | [1.696, 1.784] | -1.134 | 0.053 | 10435 |
| Diversity-only | 1.797 | [1.772, 1.822] | -1.192 | 0.373 | 10708 |
| PSRO-lite | 1.811 | [1.811, 1.811] | -1.200 | 0.333 | 1000 |

- `passed`: false
- `framing`: diagnostic
- `beats_ppo`: false
- `beats_diversity_only`: false

The strategy loop usually selects `aggressive` (seeds 0, 2, 3, 4) or a latent
(seed 1). It is slightly less exploitable than diversity-only, but the CIs
overlap, and PPO is strictly lower. Train Nash is 0.746; no method is close.

Figures: `outputs/public/paper/figures/gate2_summary.png` and
`outputs/public/paper/figures/exploitability_vs_steps.png`.

## Harness fixes shipped this cycle

- `defender_period` in `GridworldV2Config` and the exact DP transition.
- `compute_exact: true` on the paper v2 config; `timeout_penalty_weight` in
  selection so a timeout-only candidate loses to any scorer.
- PPO returns `result["_policy"]`; `policy_from_actor_critic` for exact BR.
- Stochastic-attacker BR for the PSRO meta-Nash mix.
- Real `env_steps` on the strategy loop and diversity-only.
- v2 train-layout detection via `TRAIN_LAYOUT_V2_NAME`.
- Ablation aggregation of `buffer_diversity` and `strategy_entropy`.

## Ablations (sampled $G$, 2 iterations, 5 seeds)

From `outputs/public/ablations_v2/summary.json`. Aggregation now reports
`buffer_diversity` and `strategy_entropy` correctly (previously the summary
showed 0.0 diversity). Goal rate is 0.0 on every ablation: the short
sampled-selection loop still picks a caught script.

| Ablation | Return | Buffer diversity | Strategy entropy |
| --- | ---: | ---: | ---: |
| `ebm_langevin` | -1.05 | 2.23 | 0.55 |
| `gaussian_sampler` | -1.05 | 2.29 | 0.55 |
| `no_buffer_positives` | -1.05 | 1.17 | 0.28 |
| `average_value_selection` | -1.00 | 2.74 | 0.69 |
| `candidate_count_4` | -1.05 | 0.00 | 0.00 |
| `candidate_count_12` | -1.05 | 3.61 | 0.69 |
| `opponent_samples_2` | -1.05 | 2.23 | 0.55 |
| `no_world_model` | -1.05 | 2.23 | 0.55 |

Sampler and world-model switches do not change return. Candidate count changes
diversity, not scoring.

## Learning study (5 seeds, PPO 4096, strategy 3 iterations)

From `outputs/public/learning_study_v2/summary.json`:

| Method | Return | Goal rate | Catch | Timeout | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| PPO | -0.40 | 0.00 | 0.00 | 1.00 | `ppo_policy` |
| Strategy loop | -1.05 | 0.00 | 1.00 | 0.00 | `aggressive` 0.6 / `latent` 0.4 |

PPO times out against the scripted greedy defender (`ppo_learning_weak`).
The strategy loop is caught in five steps (`strategy_execution_weak`).
Neither scores. Buffer diversity is 1.01 for the strategy loop (non-zero;
the EBM is sampling).

## Framing decision

Keep the paper diagnostic. The cycle's reportable result is not a transfer
win. It is that (1) an unwinnable game made the first Gate 2 uninterpretable,
and (2) after making the game winnable and measuring exact exploitability,
robustness-aware selection still does not produce a low-exploitability
attacker.

Future work (not this cycle): self-play PPO, world-model best response,
multi-evader pursuit.
