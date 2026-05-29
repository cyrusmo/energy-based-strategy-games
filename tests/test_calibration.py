from examples.calibrate_device import calibrate_devices


def test_calibration_cpu_smoke() -> None:
    result = calibrate_devices(warmup=0, runs=1)
    assert result["schema_version"] == "device_calibration/v1"
    jobs = result["jobs"]
    assert {item["job"] for item in jobs} >= {"langevin_sampling", "ebm_update", "policy_rollout", "ppo_update"}
    assert all(item["cpu_ms"] is not None for item in jobs)
    assert all(item["recommended_device"] in {"cpu", "mps"} for item in jobs)
