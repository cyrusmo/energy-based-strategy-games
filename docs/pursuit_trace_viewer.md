# Pursuit Trace Viewer

The pursuit trace viewer is an optional public demo surface for inspecting
multi-agent pursuit/evasion rollouts.

This viewer is intended for inspecting environment dynamics, scripted policy
behavior, and trace-level metrics. It does not demonstrate learned robustness,
optimality, or exact game-theoretic guarantees.

## What It Shows

- A validated `PursuitTrace` artifact with schema version `pursuit_trace/v1`.
- A custom 9x9 multi-evader pursuit/evasion gridworld.
- Scripted pursuer and evader behavior.
- Per-step actions, positions, rewards, captures, and active evaders.
- Summary metrics such as capture rate, survival rate, per-agent returns, and
  termination reason.

## Export A Trace

```bash
python examples/export_pursuit_trace.py --config configs/demo/custom_2_evader_9x9.yaml
```

This writes:

- `outputs/public/pursuit_demo/trace.json`
- `outputs/public/pursuit_demo/summary.json`
- `outputs/public/pursuit_demo/trajectory.png`

Generated demo artifacts are ignored by default so curated outputs can be
promoted deliberately.

## Launch The Viewer

The viewer uses Streamlit as an optional dependency:

```bash
pip install -e ".[dev,demo]"
streamlit run examples/pursuit_trace_viewer.py
```

The viewer can load a saved `trace.json`. Live rollout mode is secondary and uses
the same trace schema internally.

## Policy Comparison Diagnostics

The pursuit demo also includes a conservative empirical-game diagnostic over
scripted policies:

```bash
python examples/compare_pursuit_policies.py --config configs/demo/pursuit_policy_comparison.yaml
```

This writes:

- `outputs/public/pursuit_demo/policy_comparison.json`
- `outputs/public/pursuit_demo/policy_comparison.csv`

The JSON artifact is canonical. The CSV is a flat public summary table with one
row per pursuer-policy / evader-policy pair.

The comparison treats the matrix as rectangular and role-specific: rows are
pursuer policies, columns are evader policies, and the primary payoff is
`mean_pursuer_return`. It does not assume the game is zero-sum; evader returns
are reported separately.

The empirical-game block reports:

- payoff against a uniform column-policy mixture
- worst-case row payoff
- empirical regret against the uniform column mixture
- maximin row policy
- a payoff-weighted row-policy ranking distribution

The ranking distribution uses a multiplicative-weights-style soft scoring rule
over row payoffs. It is a ranking distribution, not an equilibrium solver.

## Trace Schema Overview

Top-level fields:

- `schema_version`: currently `pursuit_trace/v1`
- `trace_type`: currently `pursuit_evasion`
- `env_id`
- `episode_id`
- `seed`
- `grid_size`: `[height, width]`
- `num_evaders`
- `num_pursuers`
- `metadata`
- `steps`
- `summary`

Each step stores post-transition positions, roles, action labels, instantaneous
`step_rewards`, capture events, active evaders, and a `done` flag.

The summary stores termination reason, capture/survival rates, all-captured
status, mean and per-agent returns, initial/final positions, total steps, and
per-evader status.

## Current Scope

The first implementation is deliberately simple: deterministic reset, scripted
policies, simultaneous movement, boundary clipping, capture after movement, and
timeout. It does not include learning integration, obstacles, partial
observability, communication, or PettingZoo trace adaptation.
