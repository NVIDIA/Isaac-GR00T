from __future__ import annotations

import json
import random
import threading
import time
from types import SimpleNamespace
from typing import Any

from gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual import (
    BimanualToppraRolloutConfig,
    CartesianLimits,
)
from gr00t.eval.real_robot.TOPPRA.kion_client.client import KionTopics
from gr00t.eval.real_robot.TOPPRA.speed_rl import client as speed_rl_client
from gr00t.eval.real_robot.TOPPRA.speed_rl.agent import (
    SpeedRLAgent,
    speed_planner_configs,
    speedtuning_reward,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.baseline import verify_frozen_baseline
from gr00t.eval.real_robot.TOPPRA.speed_rl.client import (
    EpisodeState,
    SpeedRLKionClient,
    _parse_args,
    _speed_values_from_args,
    _validate_runtime_args,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.config import (
    BASELINE_SPEED_VALUES,
    TOPPRA_SPEED_VALUES,
    RainbowConfig,
    speed_grid,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.contract import (
    FeatureContract,
    extract_candidate_features,
    pool_candidate_feature,
    trim_candidate_features,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.interpolation import (
    InterpolatedBimanualTrajectory,
    InterpolationSpeedRLAgent,
    accelerated_sample_phases,
    resample_euler_delta_chunk,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.learner import LearnerProcess, RainbowLearner, SpeedActor
from gr00t.eval.real_robot.TOPPRA.speed_rl.network import (
    DuelingC51Network,
    NoisyLinear,
    project_categorical_distribution,
    select_masked_action,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.phases import (
    CalibrationManager,
    EpisodeOutcome,
    GreedyAcceptance,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.probe_server import build_kion_probe_observation
from gr00t.eval.real_robot.TOPPRA.speed_rl.replay import (
    PrioritizedReplayBuffer,
    Transition,
    build_n_step_transitions,
    build_rainbow_replay_transitions,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.safety import SpeedViolationMonitor
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch


FEATURE_DIM = 8
CONTRACT = FeatureContract(
    feature_dim=FEATURE_DIM,
    model_id="synthetic-vla",
    layer="action_decoder.hidden",
)


def synthetic_bandit_episodes(
    feature_dim: int,
    count: int,
    *,
    seed: int = 0,
) -> list[list[Transition]]:
    rng = np.random.default_rng(seed)
    mask = np.ones(4, dtype=np.bool_)
    episodes: list[list[Transition]] = []
    for index in range(count):
        best_action = index % 4
        state = rng.normal(0.0, 0.02, feature_dim).astype(np.float32)
        state[best_action] += 1.0
        action = (index // 4) % 4
        reward = 100.0 if action == best_action else 0.0
        episodes.append(
            [
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state.copy(),
                    done=True,
                    action_mask=mask.copy(),
                    next_action_mask=mask.copy(),
                )
            ]
        )
    return episodes


def _base_limits() -> CartesianLimits:
    return CartesianLimits(
        max_linear_velocity=(1.0,) * 3,
        max_angular_velocity=(3.0,) * 3,
        max_linear_acceleration=(5.0,) * 3,
        max_angular_acceleration=(15.0,) * 3,
        safety_margin=1.0,
    )


def _transition(action: int, reward: float = 1.0, done: bool = False) -> Transition:
    state = np.full(FEATURE_DIM, action, dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    return Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=state + 1.0,
        done=done,
        action_mask=mask,
        next_action_mask=mask,
    )


def test_frozen_toppra_and_kion_baseline_hashes() -> None:
    assert len(verify_frozen_baseline()) == 7


def test_dueling_c51_shape_running_norm_noisy_heads_and_arbitrary_mask() -> None:
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    beta_1 = config.advance_beta(0.6, 1)
    beta_2 = config.advance_beta(beta_1, 2)
    assert beta_1 == pytest.approx(0.6 + 1e-5 * 0.4)
    assert beta_2 == pytest.approx(beta_1 + 2e-5 * (1.0 - beta_1))
    network = DuelingC51Network(config)
    logits = network(torch.randn(3, FEATURE_DIM))
    assert logits.shape == (3, 4, 121)
    assert network.states_mean.shape == (FEATURE_DIM,)
    assert network.states_std.shape == (FEATURE_DIM,)
    assert network.backbone[0].out_features == 256
    assert network.backbone[2].out_features == 256
    assert network.backbone[4].out_features == 256
    assert isinstance(network.advantage_hidden, NoisyLinear)
    assert isinstance(network.value, NoisyLinear)

    action = select_masked_action(
        network,
        np.zeros(FEATURE_DIM, dtype=np.float32),
        np.array([False, False, True, False]),
        epsilon=1.0,
        rng=random.Random(0),
    )
    assert action == 2
    with pytest.raises(ValueError, match="at least one"):
        select_masked_action(
            network,
            np.zeros(FEATURE_DIM, dtype=np.float32),
            np.zeros(4, dtype=np.bool_),
            epsilon=0.0,
        )


def test_c51_projection_preserves_probability_and_terminal_reward() -> None:
    support = torch.linspace(0.0, 260.0, 51)
    probabilities = torch.zeros(2, 51)
    probabilities[:, 10] = 1.0
    projected = project_categorical_distribution(
        probabilities,
        rewards=torch.tensor([26.0, 130.0]),
        dones=torch.tensor([False, True]),
        discounts=torch.tensor([0.99**3, 0.99**3]),
        support=support,
    )
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(2))
    expected_terminal = torch.sum(projected[1] * support)
    assert float(expected_terminal) == pytest.approx(130.0, abs=1e-4)


def test_three_step_return_and_prioritized_replay_sampling() -> None:
    episode = [
        _transition(0, reward=1.0),
        _transition(1, reward=2.0),
        _transition(2, reward=3.0),
        _transition(3, reward=4.0, done=True),
    ]
    n_step = build_n_step_transitions(episode, n_step=3, gamma=0.99)
    assert n_step[0].steps == 3
    assert n_step[0].reward == pytest.approx(1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)
    assert not n_step[0].done
    assert n_step[1].done

    combined = build_rainbow_replay_transitions(episode, n_step=3, gamma=0.99)
    assert combined[0].one_step.steps == 1
    assert combined[0].n_step.steps == 3

    replay = PrioritizedReplayBuffer(16, alpha=0.2, seed=0)
    replay.extend(combined)
    replay.update_priorities(
        np.arange(4, dtype=np.int64),
        np.array([1.0, 1.0, 1.0, 100.0]),
    )
    hits = 0
    for _ in range(200):
        hits += int(replay.sample(1, beta=0.4).indices[0] == 3)
    assert hits > 100


def test_feature_contract_and_action_trim_stay_aligned() -> None:
    actions = {"left_delta_tcp": np.zeros((2, 6, 6), dtype=np.float32)}
    features = np.arange(2 * 6 * FEATURE_DIM, dtype=np.float32).reshape(
        2,
        6,
        FEATURE_DIM,
    )
    response = (
        actions,
        {
            "speed_rl": {
                "action_features": features,
                "contract": CONTRACT.to_dict(),
            }
        },
    )
    extracted = extract_candidate_features(response, CONTRACT, (6, 6))
    trimmed = trim_candidate_features(extracted, 2)
    assert [item.shape for item in trimmed] == [(4, FEATURE_DIM), (4, FEATURE_DIM)]
    np.testing.assert_array_equal(trimmed[1][0], features[1, 2])
    np.testing.assert_allclose(pool_candidate_feature(trimmed[0]), features[0, 2:].mean(axis=0))

    bad_response = (
        actions,
        {
            "speed_rl": {
                "action_features": features.astype(np.float64),
                "contract": CONTRACT.to_dict(),
            }
        },
    )
    with pytest.raises(ValueError, match="dtype"):
        extract_candidate_features(bad_response, CONTRACT, (6, 6))


def test_speedtuning_reward_keeps_speed_term_and_adds_strict_terminal_100() -> None:
    failure = speedtuning_reward(4.0, 10.0 / 30.0, terminal=True, safe_success=False)
    success = speedtuning_reward(4.0, 10.0 / 30.0, terminal=True, safe_success=True)
    nonterminal = speedtuning_reward(4.0, 10.0 / 30.0, terminal=False, safe_success=True)
    assert failure == pytest.approx((1.6, 1.6, 0.0, 10.0))
    assert success == pytest.approx((101.6, 1.6, 100.0, 10.0))
    assert nonterminal == pytest.approx(failure)


def test_speed_planners_use_exact_unmargined_constraints() -> None:
    assert speed_grid(0.7, 1.6, 0.3) == TOPPRA_SPEED_VALUES
    assert speed_grid(1.0, 4.0, 0.5) == BASELINE_SPEED_VALUES
    base = BimanualToppraRolloutConfig(
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    configs = speed_planner_configs(base)
    for config, scale in zip(configs, (0.7, 1.0, 1.3, 1.6), strict=True):
        np.testing.assert_allclose(
            config.left_limits.velocity,
            np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]) * scale,
        )
        np.testing.assert_allclose(
            config.right_limits.acceleration,
            np.array([5.0, 5.0, 5.0, 15.0, 15.0, 15.0]) * scale**2,
        )
        assert config.left_limits.safety_margin == 1.0


def _integrated_relative_path(actions: np.ndarray) -> tuple[np.ndarray, list[Rotation]]:
    positions = []
    rotations = []
    position = np.zeros(3)
    rotation = Rotation.identity()
    for action in actions:
        position = position + action[:3]
        rotation = Rotation.from_euler("xyz", action[3:]) * rotation
        positions.append(position.copy())
        rotations.append(rotation)
    return np.asarray(positions), rotations


def _speedtuning_reference_resample(values: np.ndarray, speed: float) -> np.ndarray:
    """Reference formula from SpeedTuning's public ``interpolate_action_chunk``."""

    sample_times = np.arange(0.0, len(values), speed)
    lower = np.floor(sample_times).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    fraction = (sample_times - lower)[:, None]
    return values[lower] + fraction * (values[upper] - values[lower])


def test_interpolation_baseline_resamples_incremental_euler_actions_in_se3() -> None:
    actions = np.array(
        [
            [0.01, 0.00, 0.00, 0.10, -0.05, 0.20],
            [0.00, 0.02, 0.00, -0.03, 0.04, 0.15],
            [0.00, 0.00, 0.03, 0.02, 0.01, -0.10],
        ],
        dtype=np.float64,
    )
    resampled = resample_euler_delta_chunk(actions, 1.0)
    original_positions, original_rotations = _integrated_relative_path(actions)
    actual_positions, actual_rotations = _integrated_relative_path(resampled)
    np.testing.assert_allclose(actual_positions, original_positions, atol=1e-12)
    for actual, expected in zip(actual_rotations, original_rotations, strict=True):
        assert np.linalg.norm((actual * expected.inv()).as_rotvec()) < 1e-12

    accelerated = resample_euler_delta_chunk(actions, 1.5)
    accelerated_positions, _ = _integrated_relative_path(accelerated)
    np.testing.assert_allclose(
        accelerated_positions,
        _speedtuning_reference_resample(original_positions, 1.5),
        atol=1e-12,
    )

    crossing = np.zeros((2, 6), dtype=np.float64)
    crossing[:, 5] = np.deg2rad([170.0, 20.0])
    slowed = resample_euler_delta_chunk(crossing, 0.5)
    _, slowed_rotations = _integrated_relative_path(slowed)
    assert np.linalg.norm(slowed_rotations[1].as_rotvec()) == pytest.approx(np.pi)
    assert np.linalg.norm(slowed_rotations[-1].as_rotvec()) == pytest.approx(np.deg2rad(170.0))


def test_interpolation_baseline_horizon_and_30hz_hold_semantics() -> None:
    expected_horizons = {0.7: 58, 1.0: 40, 1.3: 31, 1.6: 25}
    for speed, expected in expected_horizons.items():
        phases = accelerated_sample_phases(40, speed)
        assert len(phases) == expected
        assert phases[0] == 1.0
        assert phases[-1] == pytest.approx(min(1.0 + (expected - 1) * speed, 40.0))

    receding_horizon = accelerated_sample_phases(40, 4.0, sample_count=10)
    np.testing.assert_allclose(receding_horizon, 1.0 + np.arange(10) * 4.0)
    with pytest.raises(ValueError, match="exceed"):
        accelerated_sample_phases(40, 4.5, sample_count=10)

    config = BimanualToppraRolloutConfig(
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    left = np.array(
        [
            [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    trajectory = InterpolatedBimanualTrajectory(
        config=config,
        sequence_id=3,
        target_poses={"left": left, "right": left.copy()},
        auxiliary_actions={
            config.left_pinch_key: np.zeros((2, 1)),
            config.right_pinch_key: np.zeros((2, 1)),
        },
        action_frequency=30.0,
        source_horizon=2,
        speed_scale=1.0,
        inference_started_at_s=0.0,
        created_at_monotonic_s=0.0,
    )
    samples = trajectory.sample(np.array([0.0, 0.004, 0.032, 0.034, trajectory.duration]))
    x = samples[config.left_pose_output_key][:, 0]
    np.testing.assert_allclose(x, [0.1, 0.1, 0.1, 0.2, 0.2])
    assert trajectory.duration == pytest.approx(2.0 / 30.0)


def test_speed_violation_latches_on_frame_100_and_stale_masks_fast_actions() -> None:
    monitor = SpeedViolationMonitor(np.ones(6), np.ones(6))
    over = np.array([1.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    zero = np.zeros(6)
    for _ in range(99):
        status = monitor.update(over, zero, stale=False)
    assert not status.violation_latched
    status = monitor.update(over, zero, stale=False)
    assert status.violation_latched

    monitor.reset_episode()
    for _ in range(99):
        monitor.update(over, zero, stale=False)
    stale = monitor.update(over, zero, stale=True)
    assert stale.consecutive_counts[0][0] == 0
    assert stale.action_mask == (True, True, False, False)
    recovered = monitor.update(zero, zero, stale=False)
    assert recovered.action_mask == (True, True, True, True)


def test_calibration_requires_two_valid_episodes_and_explicit_approval(tmp_path) -> None:
    calibration = CalibrationManager(tmp_path / "calibration.json")
    assert calibration.fixed_action == 0
    calibration.record_episode(EpisodeOutcome(success=True, tracking_log="one.csv"), 3)
    assert calibration.fixed_action == 0
    calibration.record_episode(EpisodeOutcome(success=False, tracking_log="two.csv"), 2)
    assert calibration.state.awaiting_approval
    with pytest.raises(RuntimeError, match="approval"):
        _ = calibration.fixed_action
    calibration.approve_current_speed("reviewed tracking_report.json")
    assert calibration.fixed_action == 1

    calibration.record_episode(EpisodeOutcome(success=False, abort=True), 1)
    assert calibration.state.stopped
    with pytest.raises(RuntimeError, match="stopped"):
        _ = calibration.fixed_action

    with pytest.raises(ValueError, match="speed values"):
        CalibrationManager(
            tmp_path / "calibration.json",
            speed_values=BASELINE_SPEED_VALUES,
        )


def test_greedy_acceptance_persists_across_restarts(tmp_path) -> None:
    state_path = tmp_path / "greedy_sync.json"
    acceptance = GreedyAcceptance(state_path=state_path)
    acceptance.add(EpisodeOutcome(success=True, duration_s=20.0))
    restored = GreedyAcceptance(state_path=state_path)
    assert len(restored.outcomes) == 1
    assert restored.outcomes[0].safe_success


def test_async_runtime_gate_requires_raw_and_speed_rl_greedy_verification(tmp_path) -> None:
    common = [
        "--dry-run",
        "--feature-dim",
        str(FEATURE_DIM),
        "--feature-model-id",
        CONTRACT.model_id,
        "--feature-layer",
        CONTRACT.layer,
        "--left-twist-thresholds",
        "1,1,1,1,1,1",
        "--right-twist-thresholds",
        "1,1,1,1,1,1",
    ]
    _validate_runtime_args(_parse_args(common))
    with pytest.raises(ValueError, match="async-verification"):
        _validate_runtime_args(
            _parse_args([*common, "--inference-mode", "async", "--phase", "greedy"])
        )

    verification = tmp_path / "async.json"
    verification.write_text('{"raw_async_rollout_verified": true}', encoding="utf-8")
    _validate_runtime_args(
        _parse_args(
            [
                *common,
                "--inference-mode",
                "async",
                "--phase",
                "greedy",
                "--async-verification",
                str(verification),
            ]
        )
    )
    with pytest.raises(ValueError, match="async greedy"):
        _validate_runtime_args(
            _parse_args(
                [
                    *common,
                    "--inference-mode",
                    "async",
                    "--phase",
                    "online",
                    "--async-verification",
                    str(verification),
                ]
            )
        )


def test_interpolation_backend_has_isolated_defaults_and_sync_single_candidate_gate() -> None:
    common = [
        "--dry-run",
        "--execution-backend",
        "interpolation",
        "--left-twist-thresholds",
        "1,1,1,1,1,1",
        "--right-twist-thresholds",
        "1,1,1,1,1,1",
    ]
    args = _parse_args(common)
    _validate_runtime_args(args)
    assert args.state_root.as_posix() == "logs/kion_speed_rl_baseline/state"
    assert args.log_root.as_posix() == "logs/kion_speed_rl_baseline/episodes"
    assert args.baseline_k_skip == 10
    assert args.policy_chunk_horizon == 40
    assert _speed_values_from_args(args) == BASELINE_SPEED_VALUES

    with pytest.raises(ValueError, match="sync only"):
        _validate_runtime_args(_parse_args([*common, "--inference-mode", "async"]))
    with pytest.raises(ValueError, match="tts-samples 1"):
        _validate_runtime_args(_parse_args([*common, "--tts-samples", "4"]))
    with pytest.raises(ValueError, match="exceed"):
        _validate_runtime_args(_parse_args([*common, "--baseline-speed-max", "4.5"]))


def test_real_motion_requires_explicit_cartesian_target_guard() -> None:
    common = [
        "--left-twist-thresholds",
        "1,1,1,1,1,1",
        "--right-twist-thresholds",
        "1,1,1,1,1,1",
    ]
    with pytest.raises(ValueError, match="Real motion requires explicit"):
        _validate_runtime_args(_parse_args(common))

    _validate_runtime_args(
        _parse_args(
            [
                *common,
                "--left-workspace-bounds",
                "0,1,-1,1,0,2",
                "--right-workspace-bounds",
                "0,1,-1,1,0,2",
                "--max-target-position-error-m",
                "0.1",
                "--max-target-rotation-error-rad",
                "0.2",
            ]
        )
    )


def test_server_contract_is_discovered_and_optional_cli_values_are_assertions(monkeypatch) -> None:
    args = _parse_args(
        [
            "--dry-run",
            "--left-twist-thresholds",
            "1,1,1,1,1,1",
            "--right-twist-thresholds",
            "1,1,1,1,1,1",
        ]
    )
    _validate_runtime_args(args)
    monkeypatch.setattr(speed_rl_client, "discover_server_contract", lambda *args: CONTRACT)
    assert speed_rl_client._resolve_feature_contract(args) == CONTRACT

    args.feature_model_id = "wrong-model"
    with pytest.raises(ValueError, match="assertions"):
        speed_rl_client._resolve_feature_contract(args)


def test_kion_probe_observation_uses_server_modality_keys() -> None:
    config = {
        "video": {"modality_keys": ["head", "left", "right"]},
        "state": {
            "modality_keys": [
                "left_tcp",
                "left_wrist_force",
                "right_tcp",
                "right_wrist_force",
            ]
        },
        "language": {"modality_keys": ["annotation.human.task_description"]},
    }
    observation = build_kion_probe_observation(
        config,
        task="test task",
        batch_size=2,
        image_size=32,
    )
    assert observation["video"]["head"].shape == (2, 1, 32, 32, 3)
    assert observation["state"]["left_tcp"].shape == (2, 1, 7)
    assert observation["state"]["left_wrist_force"].shape == (2, 1, 3)
    assert observation["language"]["annotation.human.task_description"] == [
        ["test task"],
        ["test task"],
    ]


def test_synthetic_rainbow_training_converges_and_checkpoint_roundtrips(tmp_path) -> None:
    torch.set_num_threads(1)
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    learner = RainbowLearner(config, CONTRACT, seed=3)
    for episode in synthetic_bandit_episodes(FEATURE_DIM, 1024, seed=4):
        learner.add_episode(episode)
    result = learner.train_updates(300, require_online_gate=False)
    assert result["updates"] == 300

    probes = torch.eye(FEATURE_DIM, dtype=torch.float32)[:4]
    predicted = torch.argmax(learner.online.q_values(probes), dim=1)
    torch.testing.assert_close(predicted, torch.arange(4))

    path = tmp_path / "speed_rl.pt"
    learner.save_checkpoint(path)
    restored = RainbowLearner(config, CONTRACT, seed=9)
    restored.load_checkpoint(path)
    assert restored.stats() == learner.stats()
    for expected, actual in zip(
        learner.online.parameters(),
        restored.online.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)

    incompatible = RainbowLearner(
        config,
        FeatureContract(
            feature_dim=FEATURE_DIM,
            model_id="other-model",
            layer=CONTRACT.layer,
        ),
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        incompatible.load_checkpoint(path)

    interpolation = RainbowLearner(
        RainbowConfig(
            feature_dim=FEATURE_DIM,
            action_count=len(BASELINE_SPEED_VALUES),
            support_max=500.0,
        ),
        CONTRACT,
        execution_backend="interpolation",
    )
    with pytest.raises(ValueError, match="execution backend"):
        interpolation.load_checkpoint(path)

    changed_speeds = RainbowLearner(
        config,
        CONTRACT,
        speed_values=(0.8, 1.1, 1.4, 1.7),
    )
    with pytest.raises(ValueError, match="speed values"):
        changed_speeds.load_checkpoint(path)


@pytest.mark.serial
def test_spawn_learner_process_replay_and_checkpoint(tmp_path) -> None:
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    actor = SpeedActor(config, seed=0)
    checkpoint = tmp_path / "spawn.pt"
    with LearnerProcess(config, CONTRACT, response_timeout_s=30.0) as learner:
        assert learner.add_episode([_transition(0, done=True)]) == 1
        assert learner.stats()["replay_size"] == 1
        assert learner.install_actor_weights(actor) == 0
        learner.save(checkpoint)
    assert checkpoint.exists()


class _DelayedFeaturePolicy:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.started = threading.Event()

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del observation, options
        self.started.set()
        time.sleep(self.delay_s)
        horizon = 40
        left = np.zeros((1, horizon, 6), dtype=np.float32)
        right = np.zeros_like(left)
        left[..., 0] = 0.003
        right[..., 1] = -0.003
        pinch = np.zeros((1, horizon, 1), dtype=np.float32)
        return (
            {
                "left_delta_tcp": left,
                "right_delta_tcp": right,
                "left_pinch": pinch,
                "right_pinch": pinch.copy(),
            },
            {
                "speed_rl": {
                    "action_features": np.ones(
                        (1, horizon, FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "contract": CONTRACT.to_dict(),
                }
            },
        )

    def close(self) -> None:
        return None


def _observation() -> dict[str, np.ndarray]:
    return {
        "observation.state.left_tcp": np.array(
            [0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "observation.state.right_tcp": np.array(
            [0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    }


def _wait_for_generation(agent: SpeedRLAgent, generation: int) -> None:
    deadline = time.perf_counter() + 3.0
    while agent.diagnostics()["completed_generation"] < generation:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Speed-RL fake policy did not complete")
        time.sleep(0.002)


@pytest.mark.parametrize("delay_s", (0.1, 0.3))
def test_fake_server_latency_keeps_act_below_control_budget(delay_s: float) -> None:
    policy = _DelayedFeaturePolicy(delay_s)
    actor = SpeedActor(RainbowConfig(feature_dim=FEATURE_DIM), seed=0)
    agent = SpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    observation = _observation()
    try:
        agent.act(observation)
        assert policy.started.wait(timeout=1.0)
        durations_ms = []
        while agent.diagnostics()["completed_generation"] < 1:
            # Thread CPU time excludes unrelated host preemption on shared CI workers.
            started_ns = time.thread_time_ns()
            agent.act(observation)
            durations_ms.append((time.thread_time_ns() - started_ns) * 1e-6)
            time.sleep(0.0002)
        assert np.percentile(durations_ms, 99) < 1.0
        assert max(durations_ms) < 4.0
        assert len(agent.trajectory_decisions) == 1
        pending_decision = next(iter(agent.trajectory_decisions.values()))
        assert pending_decision["source_horizon"] == 40
        assert pending_decision["execution_horizon"] == 30
        assert pending_decision["discarded_source_actions"] == 10
        assert agent.diagnostics()["speed_rl_activated_decisions"] == 0

        agent.act(observation)
        assert agent.diagnostics()["speed_rl_activated_decisions"] == 1
        transitions = agent.finish_episode(EpisodeOutcome(success=True))
        assert len(transitions) == 1
        records = agent.last_episode_decisions()
        assert len(records) == 1
        assert "feature" not in records[0]
        assert records[0]["sequence_id"] == agent.diagnostics()["active_sequence_id"]
        assert transitions[0].reward == pytest.approx(100.0)
        assert records[0]["terminal_bonus"] == 100.0
    finally:
        agent.teardown()


def test_interpolation_agent_reuses_speed_rl_transition_lifecycle() -> None:
    policy = _DelayedFeaturePolicy(0.01)
    actor = SpeedActor(
        RainbowConfig(
            feature_dim=FEATURE_DIM,
            action_count=len(BASELINE_SPEED_VALUES),
            support_max=500.0,
        ),
        seed=0,
    )
    agent = InterpolationSpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
        baseline_action_frequency=30.0,
    )
    agent.set_fixed_action(6)
    try:
        agent.act(_observation())
        _wait_for_generation(agent, 1)
        decision = next(iter(agent.trajectory_decisions.values()))
        assert decision["execution_backend"] == "interpolation"
        assert decision["source_horizon"] == 40
        assert decision["resampled_horizon"] == 10
        assert decision["execution_horizon"] == 10
        assert decision["full_resampled_horizon"] == 10
        assert decision["last_source_action_index"] == 36.0
        assert decision["action_frequency_hz"] == 30.0

        agent.act(_observation())
        while agent.diagnostics()["active_buffer_size"]:
            agent.act(_observation())
        diagnostics = agent.diagnostics()
        assert diagnostics["speed_rl_execution_backend"] == "interpolation"
        assert diagnostics["speed_rl_activated_decisions"] == 1
        transitions = agent.finish_episode(EpisodeOutcome(success=True))
        assert len(transitions) == 1
        assert transitions[0].action == 6
        assert transitions[0].reward == pytest.approx(101.6)
        record = agent.last_episode_decisions()[0]
        assert record["executed_30hz_steps"] == pytest.approx(10.0)
        assert record["speed_reward"] == pytest.approx(1.6)
        assert record["terminal_bonus"] == 100.0
    finally:
        agent.teardown()


def test_control_fault_episode_is_logged_but_excluded_from_replay() -> None:
    policy = _DelayedFeaturePolicy(0.01)
    actor = SpeedActor(RainbowConfig(feature_dim=FEATURE_DIM), seed=0)
    agent = SpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    try:
        agent.act(_observation())
        _wait_for_generation(agent, 1)
        agent.act(_observation())
        transitions = agent.finish_episode(EpisodeOutcome(success=False, control_fault=True))
        assert transitions == []
        assert agent.last_episode_decisions()[0]["excluded_from_replay"] is True
    finally:
        agent.teardown()


def test_greedy_client_persists_labeled_episode_without_mutating_replay(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeObservations:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeResource:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def wait_for_connection(self, _timeout: float) -> None:
            return None

        def configure(self, _timeout: float) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakeRecorder(FakeResource):
        def __init__(self, directory, _metadata) -> None:
            super().__init__()
            self.directory = directory
            self.directory.mkdir(parents=True)
            self.csv_path = self.directory / "tcp_tracking.csv"
            self.csv_path.touch()

    class FakeLearner:
        def __init__(self) -> None:
            self.add_calls = 0
            self.save_calls = 0
            self.snapshot = {
                "replay_size": 320,
                "action_counts": (80, 80, 80, 80),
                "policy_version": 7,
                "update_count": 22,
                "activated_decisions": 320,
                "online_training_ready": True,
                "beta": 0.6,
            }

        def stats(self) -> dict[str, Any]:
            return dict(self.snapshot)

        def install_actor_weights(self, actor: Any) -> int:
            actor.policy_version = self.snapshot["policy_version"]
            return actor.policy_version

        def add_episode(self, _transitions: Any) -> int:
            self.add_calls += 1
            return 1

        def save(self, _path: Any) -> dict[str, Any]:
            self.save_calls += 1
            return self.stats()

    class FakeAgent:
        def __init__(self) -> None:
            self.config = SimpleNamespace(inference_mode="sync")
            self.feature_contract = CONTRACT
            self.speed_values = TOPPRA_SPEED_VALUES
            self.fixed_action = None
            self.greedy = False
            self.action_mask = None
            self.closed = False

        def reset_execution(self) -> None:
            return None

        def set_fixed_action(self, value: int | None) -> None:
            self.fixed_action = value

        def set_greedy(self, value: bool) -> None:
            self.greedy = value

        def set_action_mask(self, value: Any) -> None:
            self.action_mask = tuple(value)

        def finish_episode(self, _outcome: EpisodeOutcome) -> list[Transition]:
            return [_transition(0, reward=1.0, done=True)]

        def last_episode_decisions(self) -> list[dict[str, Any]]:
            return [{"sequence_id": 1, "action_index": 0, "speed_scale": 0.7}]

        def diagnostics(self) -> dict[str, Any]:
            return {
                "speed_rl_activated_decisions": 320,
                "speed_rl_policy_version": 7,
            }

        def teardown(self) -> None:
            self.closed = True

    monkeypatch.setattr(speed_rl_client, "RolloutObservationBuffer", FakeObservations)
    monkeypatch.setattr(speed_rl_client, "KionDualArmServo", FakeResource)
    monkeypatch.setattr(speed_rl_client, "KionPinchExecutor", FakeResource)
    monkeypatch.setattr(speed_rl_client, "TrackingRecorder", FakeRecorder)

    calibration = CalibrationManager(tmp_path / "state" / "calibration.json")
    calibration.state.complete = True
    calibration.state.speed_index = 3
    learner = FakeLearner()
    agent = FakeAgent()
    actor = SimpleNamespace(policy_version=0)
    client = SpeedRLKionClient(
        ros_types=SimpleNamespace(rospy=object(), hand_namespace="fake.hand.msg"),
        topics=KionTopics(),
        agent=agent,
        actor=actor,
        learner=learner,
        safety=SpeedViolationMonitor(np.ones(6), np.ones(6)),
        calibration=calibration,
        phase="greedy",
        checkpoint_path=tmp_path / "speed_rl.pt",
        log_root=tmp_path / "episodes",
        control_frequency=250,
        servo_gain=800,
        startup_timeout_s=1,
        max_state_age_s=1,
        max_image_age_s=1,
        task="parcel",
        dry_run=True,
    )
    client._online_budget.completed_episodes = 100
    client._twist_stale = False
    client._start_episode()
    assert client.state is EpisodeState.RUNNING
    assert agent.greedy
    assert agent.action_mask == (True, True, True, True)
    episode_directory = client.recorder.directory
    client._finish_episode(success=True, reason="operator success")
    client._finalizer.join(timeout=2)

    assert learner.add_calls == 0
    assert learner.save_calls == 0
    outcome = json.loads((episode_directory / "outcome.json").read_text())
    assert outcome["finalization_complete"] is True
    assert outcome["safe_success"] is True
    assert outcome["decisions"][0]["sequence_id"] == 1
    assert (
        GreedyAcceptance(state_path=tmp_path / "state" / "greedy_sync_acceptance.json").result()[
            "episodes"
        ]
        == 1
    )
    with pytest.raises(RuntimeError, match="finish and reset first"):
        client._start_episode()
    client._begin_reset()
    assert client.state is EpisodeState.RESETTING
    client._confirm_reset_ready()
    assert client.state is EpisodeState.IDLE
    client.close()


def test_reset_execution_invalidates_late_policy_result() -> None:
    policy = _DelayedFeaturePolicy(0.1)
    actor = SpeedActor(RainbowConfig(feature_dim=FEATURE_DIM), seed=0)
    agent = SpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    try:
        agent.act(_observation())
        assert policy.started.wait(timeout=1.0)
        agent.reset_execution()
        time.sleep(0.2)
        assert agent.diagnostics()["ready_sequence_id"] is None
        assert agent.trajectory_decisions == {}
        assert agent.diagnostics()["speed_rl_episode_epoch"] == 1
    finally:
        agent.teardown()
