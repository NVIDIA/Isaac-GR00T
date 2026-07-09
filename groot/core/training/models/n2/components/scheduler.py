# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow Matching Schedulers — training + inference, no diffusers dependency.

BaseFlowMatchScheduler defines the shared API.  Two concrete schedulers:
- FlowMatchUniPCScheduler: shifted-sigma schedule + UniPC solver (for video)
- BetaFlowMatchEulerScheduler: Beta-distribution sampling + Euler solver (for action, N1.7-style)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Beta


def _to_device_non_blocking_from_cpu(
    tensor: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = torch.device(device)
    if tensor.device.type == "cpu" and target_device.type == "cuda":
        tensor = tensor.pin_memory()
        return tensor.to(device=target_device, dtype=dtype, non_blocking=True)
    return tensor.to(device=target_device, dtype=dtype)


# ─── Base class ───────────────────────────────────────────────────────────────


class BaseFlowMatchScheduler(ABC):
    """Common interface for flow-matching schedulers."""

    @abstractmethod
    def setup_training(self, num_timestep_buckets: int = 1000) -> None: ...

    @abstractmethod
    def sample_timesteps(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noise levels for training.

        Returns:
            (timesteps, sigmas) each (B, 1).
            - timesteps: scheduler-native value (used by add_noise / training_weight).
            - sigmas: always in sigma convention (0=clean, 1=noise) for transformer AdaLN.
        """

    @abstractmethod
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def training_target(self, sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, **kwargs) -> None: ...

    @abstractmethod
    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor,
             step_index: int) -> tuple: ...


# ─── Video scheduler: shifted-sigma + UniPC ──────────────────────────────────


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    solver_order: int = 2
    prediction_type: str = "flow_prediction"
    shift: float = 1.0
    predict_x0: bool = True
    solver_type: str = "bh2"
    lower_order_final: bool = True
    disable_corrector: list[int] | None = None
    final_sigmas_type: str = "zero"


class FlowMatchUniPCScheduler(BaseFlowMatchScheduler):
    """Unified scheduler for flow-matching diffusion: training noise + UniPC inference."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

        self.num_inference_steps = None
        self.predict_x0 = config.predict_x0

        # Build full training schedule
        alphas = np.linspace(1, 1 / config.num_train_timesteps, config.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).float()
        sigmas = config.shift * sigmas / (1 + (config.shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * config.num_train_timesteps

        self._reset_state()

    def _reset_state(self):
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = None

    # ─── Training Methods ─────────────────────────────────────────────────────

    def setup_training(self, num_timestep_buckets: int = 1000):
        """Prepare training schedule with Gaussian loss weighting."""
        x = self.timesteps[:num_timestep_buckets]
        y = torch.exp(-2 * ((x - num_timestep_buckets / 2) / num_timestep_buckets) ** 2)
        y_shifted = y - y.min()
        self._training_weights = y_shifted * (num_timestep_buckets / y_shifted.sum())

    @torch.compiler.disable()
    def sample_timesteps(self, batch_size, device, dtype=torch.float32):
        with torch.amp.autocast(dtype=torch.float32, device_type="cuda" if "cuda" in str(device) else "cpu"):
            ids = torch.randint(0, self.config.num_train_timesteps, (batch_size, 1))
            timesteps = _to_device_non_blocking_from_cpu(self.timesteps[ids], device)
            sigmas = _to_device_non_blocking_from_cpu(self.sigmas[ids], device)
        return timesteps, sigmas

    @torch.compiler.disable()
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Add noise: x_t = (1 - sigma) * x_0 + sigma * noise."""
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0).cpu()).abs(), dim=0
        )
        sigma = self.sigmas[timestep_id].to(device=original_samples.device, dtype=original_samples.dtype)
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Velocity prediction target: v = noise - x_0."""
        return noise - sample

    @torch.compiler.disable()
    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Gaussian-weighted per-timestep loss weight."""
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0).to(self.timesteps.device)).abs(), dim=0
        )
        return self._training_weights[timestep_id]

    # ─── Inference Methods ────────────────────────────────────────────────────

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, shift: float | None = None):
        """Set discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        if shift is None:
            shift = self.config.shift

        sigma_max = self.sigmas[0].item()
        sigma_min = self.sigmas[-1].item()
        sigmas = np.linspace(sigma_max, sigma_min, num_inference_steps + 1).copy()[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        sigma_last = 0 if self.config.final_sigmas_type == "zero" else sigma_min
        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self._reset_state()

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma):
        return 1 - sigma, sigma

    def convert_model_output(self, model_output: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        """Convert velocity prediction to x0 prediction."""
        sigma_t = self.sigmas[step_index]
        if self.predict_x0:
            return sample - sigma_t * model_output
        else:
            return sample - (1 - sigma_t) * model_output

    def multistep_uni_p_bh_update(self, model_output: torch.Tensor, sample: torch.Tensor,
                                   order: int, step_index: int) -> torch.Tensor:
        """UniP predictor step (B(h) version)."""
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks, D1s = [], []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh) if self.config.solver_type == "bh2" else hh

        R, b = [], []
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device) if order == 2 else torch.linalg.solve_ex(R[:-1, :-1], b[:-1])[0].to(x.dtype)
        else:
            D1s, rhos_p = None, None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output: torch.Tensor, last_sample: torch.Tensor,
                                   this_sample: torch.Tensor, order: int, step_index: int) -> torch.Tensor:
        """UniC corrector step (B(h) version)."""
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks, D1s = [], []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh) if self.config.solver_type == "bh2" else hh

        R, b = [], []
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        D1s = torch.stack(D1s, dim=1) if len(D1s) > 0 else None
        rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device) if order == 1 else torch.linalg.solve_ex(R, b)[0].to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * (model_t - m0))
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * (model_t - m0))

        return x_t.to(x.dtype)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor,
             step_index: int, return_dict: bool = False) -> tuple:
        """One denoising step with UniPC predictor-corrector."""
        disable_corrector = self.config.disable_corrector or []
        use_corrector = step_index > 0 and step_index - 1 not in disable_corrector and self.last_sample is not None

        model_output_convert = self.convert_model_output(model_output, sample, step_index)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(model_output_convert, self.last_sample, sample, self.this_order, step_index).clone()

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(model_output, sample, self.this_order, step_index).clone()

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        return (prev_sample,)


# ─── Action scheduler: Beta distribution + Euler ─────────────────────────────


@dataclass
class BetaSchedulerConfig:
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    # Sign convention for the velocity target.
    # "x0_minus_noise" (default) — v = x_0 − noise. Matches the action head's
    # existing training; back-compat with any model trained under this
    # convention, including all currently-saved action-module weights.
    # "noise_minus_x0" — v = noise − x_0. Matches FlowMatchUniPCScheduler.
    # Use when posttraining a UniPC-pretrained model with the Beta schedule
    # so the existing model output direction aligns with the new target.
    velocity_direction: str = "x0_minus_noise"
    # Inference grid. False (default) → uniform-t grid, fixed dt = 1/N (matches
    # action-head training). True → DreamZero recipe: linspace(σ_max, σ_min, N+1)
    # then σ-shift (= shift·σ / (1 + (shift−1)·σ)) with final σ=0; variable
    # dt per step. The shifted grid concentrates queries at high σ, matching
    # Beta(α, β)-trained models whose training mass lives there. Use True for
    # video heads trained with Beta + α≫β (e.g. α=7).
    use_shifted_inference_grid: bool = False


class BetaFlowMatchEulerScheduler(BaseFlowMatchScheduler):
    """N1.7-style flow matching: Beta(alpha, beta) noise sampling + Euler inference.

    Convention: t ∈ [0, 1] where t=1 is clean, t=0 is pure noise.
    Velocity target: controlled by config.velocity_direction.
      - "x0_minus_noise" (default): v = x_0 - noise.  Euler: x += dt * v.
      - "noise_minus_x0":            v = noise - x_0.  Euler: x -= dt * v.
    Noisy sample: x_t = (1 - t) * noise + t * x_0.

    sample_timesteps() returns (t, sigma) where sigma = 1 - t so that
    the transformer AdaLN sees consistent sigma convention (0=clean).
    """

    def __init__(self, config: BetaSchedulerConfig):
        self.config = config
        self.beta_dist = Beta(config.beta_alpha, config.beta_beta)
        self.num_inference_steps = None
        self.sigmas = None
        self.timesteps = None

    def setup_training(self, num_timestep_buckets: int = 1000):
        pass

    @torch.compiler.disable()
    def sample_timesteps(self, batch_size, device, dtype=torch.float32):
        with torch.amp.autocast(dtype=torch.float32, device_type="cuda" if "cuda" in str(device) else "cpu"):
            t = _to_device_non_blocking_from_cpu(
                self.beta_dist.sample([batch_size]),
                device,
                dtype=dtype,
            )
        t = (1 - t) * self.config.noise_s  # (B,)
        t = t.unsqueeze(1)  # (B, 1)
        sigma = 1.0 - t  # sigma convention for AdaLN
        return t, sigma

    @torch.compiler.disable()
    def add_noise(self, original_samples, noise, timestep):
        """x_t = (1 - t) * noise + t * x_0.

        `@torch.compiler.disable()` mirrors FlowMatchUniPCScheduler — without
        it, the `(1 - t) * noise + t * original_samples` allocations are
        captured into reduce-overhead's CUDA-graph-trees pool, and pool
        addresses diverge across replays → `curr_block->next == nullptr`
        at step 2-3. The UniPC path has always had this decorator, which is
        why only Beta-video training ever tripped the bug.
        """
        t = timestep
        while t.ndim < original_samples.ndim:
            t = t.unsqueeze(-1)
        return (1 - t) * noise + t * original_samples

    def training_target(self, sample, noise):
        """Velocity target (sign controlled by config.velocity_direction)."""
        if self.config.velocity_direction == "noise_minus_x0":
            return noise - sample
        return sample - noise

    @torch.compiler.disable()
    def training_weight(self, timestep):
        """Uniform weighting.

        `@torch.compiler.disable()` mirrors FlowMatchUniPCScheduler: keeps the
        `torch.ones_like(...)` allocation outside Inductor's compiled pool so
        reduce-overhead CUDA-graph-trees doesn't capture a per-step allocation
        address and fail `apply_checkpoint_execution_state_in_allocator`
        (`curr_block->next == nullptr`) on step 2-3. Only video-side training
        actually calls training_weight; action training bypasses it, which is
        why the historical action use-case never tripped the bug.
        """
        return torch.ones_like(timestep)

    def set_timesteps(self, num_inference_steps, device=None, shift: float = 1.0, **kwargs):
        """Build inference σ grid.

        - use_shifted_inference_grid=False (default): uniform t-grid, fixed
          dt=1/N. Sigmas span [1, 1/N] (never reach 0). Good for action heads
          trained with mild Beta concentration.
        - use_shifted_inference_grid=True: DreamZero recipe — linspace(1, ε, N+1)
          followed by σ-shift transform; final σ=0. Variable dt per step
          (= σ_next − σ_curr). Required for Beta(7,1)-trained video heads
          where the model's training mass concentrates at high σ.
        """
        self.num_inference_steps = num_inference_steps
        if self.config.use_shifted_inference_grid:
            sigma_max, sigma_min = 1.0, 0.003 / 1.002
            sigmas_np = np.linspace(sigma_max, sigma_min, num_inference_steps + 1)
            if shift != 1.0:
                sigmas_np = shift * sigmas_np / (1 + (shift - 1) * sigmas_np)
            sigmas_np[-1] = 0.0
            sigmas = torch.from_numpy(sigmas_np.astype(np.float32))
            if device is not None:
                sigmas = sigmas.to(device)
            self.sigmas = sigmas
            self.timesteps = sigmas[:-1].clone()
            self.dt = None
        else:
            self.dt = 1.0 / num_inference_steps
            # t grid: [0, dt, 2*dt, ...] — noise level decreases as t increases
            t = torch.linspace(0, 1 - self.dt, num_inference_steps, device=device)
            self.timesteps = t
            self.sigmas = 1.0 - t  # sigma convention for AdaLN

    def step(self, model_output, timestep, sample, step_index):
        """Euler integration. Sign matches config.velocity_direction; dt is
        variable when use_shifted_inference_grid=True (= σ_next − σ_curr) and
        fixed = 1/N otherwise.

        x0_minus_noise (v = x_0 − noise): step in +v direction by |dσ|
          → uniform: prev = sample + dt·v
          → shifted: prev = sample + (σ_curr − σ_next)·v
        noise_minus_x0 (v = noise − x_0): step in −v direction by |dσ|
          → uniform: prev = sample − dt·v
          → shifted: prev = sample + (σ_next − σ_curr)·v
        """
        if self.config.use_shifted_inference_grid:
            sigma_curr = self.sigmas[step_index]
            sigma_next = self.sigmas[step_index + 1]
            if self.config.velocity_direction == "noise_minus_x0":
                return (sample + (sigma_next - sigma_curr) * model_output,)
            return (sample + (sigma_curr - sigma_next) * model_output,)
        if self.config.velocity_direction == "noise_minus_x0":
            return (sample - self.dt * model_output,)
        return (sample + self.dt * model_output,)
