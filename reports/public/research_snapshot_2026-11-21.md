# Research Snapshot: PPO on Gridworld v2

Date: 2026-11-21

Status: diagnostic. Numbers come from ad hoc probe runs on the train layout
(`corridor_split`), 131,072 steps unless stated, evaluated deterministically
from the layout start against the scripted defender. They are not generated
by a committed runner.

## Summary

PPO's zero goal rate on v2 is a standoff, not a budget problem. Hardened
settings reach the goal on 11 of 15 seeds. Two principled fixes were tried
and made things worse. The hardened settings are adopted as
`configs/gridworld_v2_ppo_hardened.yaml`; the paper baseline is unchanged.

## Diagnosis

Failing policies walk toward the nearest goal, stop at the mouth of the top
wall gap at (2, 2), and wait for the timeout. The scripted defender chases
the attacker, stops when adjacent, and ends up inside the gap at (2, 3).
Stepping forward is a catch (-1); standing still is a timeout (-0.4). The
escape is to retreat and use the lower gap, which goal-distance shaping
penalises because it ignores the defender. A second variant freezes at (0, 2)
with the defender adjacent.

## Results

Seeds reaching goal rate 1.0 from the layout start.

| Setting | Seeds |
| --- | ---: |
| Default settings, 4,096 steps (paper baseline) | 0 of 3 |
| Default settings, 32,768 steps | 0 of 3 |
| Default settings, 131,072 steps | 0 of 3 |
| Default settings, batches of 1,024 | 0 of 5 |
| Tuned: shaping x10, entropy bonus 0.01, learning rate 1e-3 | 3 of 5 |
| Tuned, batches of 1,024 | 0 of 5 |
| Tuned + half of training episodes from random starts | 4 of 5, then 7 of 10 on fresh seeds |
| Tuned + all training episodes from random starts | 1 of 5 |
| Default settings + half random starts | 1 of 5 |
| Tuned + half random starts + entropy bonus annealed 0.05 to 0.005 | 3 of 5 |
| Progress shaping, entropy bonus 0.01 | 0 of 5 |
| Progress shaping, entropy annealed | 0 of 5 |
| Progress shaping, entropy annealed, half random starts | 1 of 5 |
| Progress shaping, entropy bonus 0.01, half random starts | 1 of 5 |

All progress-shaping rows use learning rate 1e-3 and `shaping_coef` 2.5, which
matches the per-step size of the tuned legacy shaping.

## What was tried and did not help

- **Progress shaping** (`shaping_mode: progress`): a non-negative potential,
  zero at episode end, that no longer pays for standing still. It removed
  every success without random starts. Interpretation, not tested: with this
  potential any episode ending near a goal without scoring forfeits the
  accumulated potential, which makes approaching the guarded gap look worse.
- **Entropy schedule** (`entropy_coef_final`): annealing from 0.05 to 0.005
  did not beat a constant 0.01 (3 of 5 against 4 of 5; too few seeds to
  separate).
- **Larger batches**: fewer updates at the same step budget; no successes.

Both options remain in the code, off by default.

## Adopted

`configs/gridworld_v2_ppo_hardened.yaml`: legacy shaping x10, entropy bonus
0.01, learning rate 1e-3, half of training episodes from random free cells,
131,072 steps. Success from the layout start on 11 of 15 seeds (4 of 5
during tuning, 7 of 10 on seeds 5-14 run once afterwards).

`configs/gridworld_v2_ppo.yaml` is unchanged, so the learning-study,
held-out and Gate 2 numbers in the paper are unaffected.

## Caveats

- About a quarter of seeds still fail, always by freezing next to the
  defender.
- Success is measured against one scripted defender on one layout. Exact
  exploitability of the hardened policy has not been computed.
- Gate 2 still uses the 4,096-step baseline. A hardened PPO could raise or
  lower its exploitability, so no claim relative to PPO should change until
  Gate 2 is rerun.

## Next

- Gate 2 was rerun with the hardened baseline on 2026-11-22; see
  `research_snapshot_2026-11-22.md`. The hardened policy is more exploitable
  (1.768 against 1.598).
- Train against a mix of scripted defenders, which removes the single
  standoff this defender creates.
