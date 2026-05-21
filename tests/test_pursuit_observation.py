import numpy as np
import pytest

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.models.pursuit_observation import (
    OBSERVATION_SCHEMA,
    PursuitObservationSpec,
    encode_evader_observation,
    encode_pursuer_observation,
    validate_pursuit_ppo_env,
)


def test_pursuit_observation_spec_metadata_and_shape() -> None:
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=1, max_evaders=2)
    env = MultiEvaderPursuitEnv()
    obs = encode_pursuer_observation(env, "pursuer_0", spec)
    metadata = spec.to_dict()

    assert metadata["observation_schema"] == OBSERVATION_SCHEMA
    assert metadata["obs_dim"] == 20
    assert len(metadata["feature_order"]) == spec.obs_dim
    assert obs.shape == (spec.obs_dim,)
    assert np.isfinite(obs).all()


def test_evader_observation_has_fixed_finite_shape() -> None:
    spec = PursuitObservationSpec(role="evader", max_pursuers=1, max_evaders=2)
    env = MultiEvaderPursuitEnv()
    obs = encode_evader_observation(env, "evader_0", spec)

    assert spec.obs_dim == 14
    assert obs.shape == (14,)
    assert np.isfinite(obs).all()


def test_pursuer_observation_masks_active_captured_and_absent_slots() -> None:
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=1, max_evaders=2)
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(5, 5),
            num_evaders=1,
            num_pursuers=1,
            max_steps=4,
            pursuer_starts=((2, 2),),
            evader_starts=((2, 3),),
            evader_goals=((4, 4),),
        )
    )

    initial = encode_pursuer_observation(env, "pursuer_0", spec)
    assert initial[8] == pytest.approx(1.0)
    assert initial[9] == pytest.approx(0.0)
    assert np.all(initial[13:20] == 0.0)

    env.step({"pursuer_0": "right", "evader_0": "stay"})
    captured = encode_pursuer_observation(env, "pursuer_0", spec)
    assert captured[8] == pytest.approx(0.0)
    assert captured[9] == pytest.approx(1.0)


def test_pursuit_ppo_observation_rejects_unsupported_agent_counts() -> None:
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=1, max_evaders=2)
    env = MultiEvaderPursuitEnv(MultiEvaderPursuitConfig(num_evaders=3, num_pursuers=1))

    with pytest.raises(ValueError, match="more evaders"):
        encode_pursuer_observation(env, "pursuer_0", spec)
    with pytest.raises(ValueError, match="more evaders"):
        validate_pursuit_ppo_env(env, spec)
