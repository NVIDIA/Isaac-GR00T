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
"""Distribution-level open-loop metrics.

These metrics complement the existing per-step ``MSE`` / ``MAE`` on action
trajectories with two trajectory-distribution-level signals:

* :func:`frechet_gaussian_over_time` — 2-Wasserstein (Bures / Fréchet)
  distance between the empirical Gaussian fits to the time distribution of
  predicted vs ground-truth action sequences. Captures structural drift
  beyond pointwise residual (e.g. a phase-shifted but otherwise valid
  trajectory has high MSE but low Fréchet).
* :func:`reverse_kl_from_samples` — reverse KL between an empirical
  KDE-style estimate of the policy distribution (from ``M`` stochastic
  samples per query) and a unit-variance Gaussian centred at the GT.
  Penalises both bias (predictive mean off-target) and over-confidence
  (samples clustered far from GT). Requires ``M >= 2`` policy samples.

Implementations are torch-native (so we can drop them straight into the
in-process eval callback's GPU forward path), but accept numpy ndarrays
at the boundary for convenience.

Both functions return *per-sample* tensors of shape ``(B,)`` so the
caller can take ``.mean()`` / ``.min()`` / ``.max()`` across a batch of
trajectory windows. We keep the internal numerics in fp32 even when the
inputs are bf16 — Fréchet requires symmetric eigendecompositions which
are unstable in low precision.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "energy_distance",
    "energy_score",
    "frechet_distance",
    "frechet_gaussian_over_time",
    "pooled_wasserstein_1d",
    "pooled_wasserstein_nd",
    "pooled_wasserstein_report",
    "reverse_kl_from_samples",
    "sliced_wasserstein",
    "video_distributional_report",
]


def _to_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


def frechet_gaussian_over_time(
    pred,
    tgt,
    *,
    squared: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample 2-Wasserstein distance between time-distributions.

    Fits an empirical Gaussian over the *T* time steps of each ``(B, T, D)``
    sample independently, then returns the Bures / Fréchet distance
    between the predicted and ground-truth Gaussians per ``B``.

    Args:
        pred: ``(B, T, D)`` (or ``(B, T, ...)`` — flattened to ``(B, T, D)``).
        tgt:  ``(B, T, D)`` matching shape with ``pred``.
        squared: If ``True`` return ``W2^2`` instead of ``W2``.
        eps: Jitter for covariance regularisation + eigenvalue clamp.

    Returns:
        ``(B,)`` tensor of W2 (or W2^2) distances. Always >= 0.

    Notes:
        For ``T == 1`` the empirical covariance is exactly zero, so the
        distance reduces to the L2 distance between the (single) means.
        The implementation is robust to that case (returns
        ``||mu_pred - mu_tgt||_2`` as expected).

        The two-step ``eigh`` is the symmetric-PSD square root chain
        from the Bures formula
        ``trace(C1) + trace(C2) - 2 * trace((C2^{1/2} C1 C2^{1/2})^{1/2})``;
        we always work in fp32 because ``eigh`` is unstable in bf16.
    """
    p = _to_torch(pred)
    t = _to_torch(tgt)
    if p.shape[:2] != t.shape[:2]:
        raise ValueError(
            f"frechet_gaussian_over_time: pred/tgt must match in (B, T); got {tuple(p.shape)} vs {tuple(t.shape)}"
        )

    B, T = p.shape[:2]
    p = p.to(torch.float32)
    t = t.to(torch.float32)
    if p.ndim > 3:
        D = int(np.prod(p.shape[2:]))
        X = p.reshape(B, T, D)
        Y = t.reshape(B, T, D)
    else:
        D = p.shape[-1]
        X, Y = p, t

    m1 = X.mean(dim=1)
    m2 = Y.mean(dim=1)

    if T <= 1:
        C1 = torch.zeros(B, D, D, device=X.device, dtype=X.dtype)
        C2 = torch.zeros(B, D, D, device=X.device, dtype=X.dtype)
    else:
        Xc = X - m1.unsqueeze(1)
        Yc = Y - m2.unsqueeze(1)
        C1 = (Xc.transpose(1, 2) @ Xc) / (T - 1)
        C2 = (Yc.transpose(1, 2) @ Yc) / (T - 1)

    eye = torch.eye(D, device=X.device, dtype=X.dtype).expand(B, D, D)
    C1 = C1 + eps * eye
    C2 = C2 + eps * eye

    w2, V2 = torch.linalg.eigh(C2)
    w2 = torch.clamp(w2, min=eps)
    C2_sqrt = V2 @ torch.diag_embed(torch.sqrt(w2)) @ V2.transpose(-1, -2)

    inner = C2_sqrt @ C1 @ C2_sqrt
    w_inner, V_inner = torch.linalg.eigh(inner)
    w_inner = torch.clamp(w_inner, min=eps)
    inner_sqrt = V_inner @ torch.diag_embed(torch.sqrt(w_inner)) @ V_inner.transpose(-1, -2)

    mean_term = (m1 - m2).pow(2).sum(dim=1)
    trace_term = (
        C1.diagonal(dim1=-2, dim2=-1).sum(-1)
        + C2.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 2.0 * inner_sqrt.diagonal(dim1=-2, dim2=-1).sum(-1)
    )
    w2_sq = torch.clamp(mean_term + trace_term, min=0.0)
    return w2_sq if squared else torch.sqrt(w2_sq)


def frechet_distance(cloud_a, cloud_b, *, squared: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Fréchet (FID/FVD) distance between two embedding clouds.

    Fits one Gaussian to *each* cloud and returns the (squared) 2-Wasserstein / Bures
    distance between them — the FID/FVD formula:
    ``||mu_a - mu_b||^2 + Tr(Sa + Sb - 2 (Sa Sb)^{1/2})``.

    This is exactly **FVD** when fed I3D *clip* features (motion-aware); on the per-frame
    DINOv3 features the open-loop eval uses by default it is an appearance-only Fréchet
    (FID-style) — the backbone, not this function, decides which. Unlike
    :func:`sliced_wasserstein` / :func:`energy_distance` it is *parametric* (Gaussian fit),
    so it is blind to multimodality — report it alongside, not instead of, those.

    Distinct from :func:`frechet_gaussian_over_time`, which fits a Gaussian over the time
    axis of one ``(B, T, D)`` trajectory; this fits over the *samples* of a cloud.

    Args:
        cloud_a: ``(N, D)`` embeddings (e.g. predicted videos).
        cloud_b: ``(M, D)`` embeddings (e.g. GT videos); ``M`` may differ from ``N``.
        squared: if ``True`` (FID/FVD convention) return ``W2^2``, else ``W2``.
        eps: covariance jitter + eigenvalue clamp (fp32 ``eigh`` stability).

    Returns:
        Scalar ``torch.Tensor`` (fp32), ``>= 0``.
    """
    a = _to_torch(cloud_a).to(torch.float32)
    b = _to_torch(cloud_b).to(torch.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"frechet_distance: clouds must be (N, D) / (M, D); got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"frechet_distance: embedding dims must match; got D={a.shape[1]} vs D={b.shape[1]}")

    D = a.shape[1]
    mu_a, mu_b = a.mean(0), b.mean(0)
    Aa, Ab = a - mu_a, b - mu_b
    eye = torch.eye(D, dtype=torch.float32, device=a.device)
    # max(n-1, 1): a single-sample cloud collapses to a point mass (cov ~= eps*I), so
    # the Fréchet distance reduces to ||mu_a - mu_b||^2 — well-defined, no div-by-zero.
    Ca = (Aa.t() @ Aa) / max(a.shape[0] - 1, 1) + eps * eye
    Cb = (Ab.t() @ Ab) / max(b.shape[0] - 1, 1) + eps * eye

    # Bures: (Ca^{1/2} Cb Ca^{1/2})^{1/2}, via two symmetric eigendecompositions.
    wa, Va = torch.linalg.eigh(Ca)
    Ca_sqrt = Va @ torch.diag(torch.sqrt(torch.clamp(wa, min=eps))) @ Va.t()
    inner = Ca_sqrt @ Cb @ Ca_sqrt
    wi, Vi = torch.linalg.eigh(inner)
    inner_sqrt = Vi @ torch.diag(torch.sqrt(torch.clamp(wi, min=eps))) @ Vi.t()

    mean_term = (mu_a - mu_b).pow(2).sum()
    trace_term = torch.trace(Ca) + torch.trace(Cb) - 2.0 * torch.trace(inner_sqrt)
    fd_sq = torch.clamp(mean_term + trace_term, min=0.0)
    return fd_sq if squared else torch.sqrt(fd_sq)


def pooled_wasserstein_1d(pred, gt, *, p: int = 1) -> torch.Tensor:
    """Per-dimension 1-D ``p``-Wasserstein between pooled empirical action dists.

    Treats each action dimension independently: for dim ``d`` it compares the
    empirical distribution of *all* predicted values ``pred[:, d]`` against all
    ground-truth values ``gt[:, d]`` — pooled across the whole eval set — with
    the timestep pairing intentionally dropped. So this is the *distributional*
    analogue of the pointwise errors: ``p=1`` → ``W₁`` (L1 ground cost,
    MAE-flavored), ``p=2`` → ``W₂`` (L2 ground cost; rooted, RMSE-scaled).
    It asks whether the policy's per-dim action *values* occupy the same
    distribution as GT, not whether they match step-for-step — so it is a
    complement to, not a replacement for, ``mse`` / ``mae`` (a phase-shifted but
    otherwise-correct policy scores low here but high on MSE).

    ``pred`` and ``gt`` must be equal-size ``(N, D)`` — one predicted and one GT
    action vector per evaluated step — which makes the 1-D ``W_p`` exact via
    order-statistic pairing:
    ``W_p(pred[:, d], gt[:, d])^p = (1/N) Σ_i |sort(pred[:, d])_i - sort(gt[:, d])_i|^p``.

    Args:
        pred: ``(N, D)`` pooled predicted action values.
        gt:   ``(N, D)`` pooled ground-truth action values (same ``N``).
        p: Wasserstein order; ``1`` (distributional MAE) or ``2`` (distributional
            MSE). Square the ``p=2`` result for an MSE-scaled quantity.

    Returns:
        ``(D,)`` tensor of per-dimension ``W_p`` (always >= 0). Callers typically
        take ``.mean()`` over dims, mirroring how ``mse``/``mae`` average dims.
    """
    if p < 1:
        raise ValueError(f"pooled_wasserstein_1d: p must be >= 1, got {p}")
    a = _to_torch(pred).to(torch.float32)
    b = _to_torch(gt).to(torch.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            "pooled_wasserstein_1d: pred/gt must be equal-size (N, D); "
            f"got {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    # Equal-size empirical 1-D W_p = order-statistic pairing (sort each marginal).
    diff = (torch.sort(a, dim=0).values - torch.sort(b, dim=0).values).abs()
    if p == 1:
        return diff.mean(dim=0)
    return diff.pow(p).mean(dim=0).pow(1.0 / p)


def _assignment_cost(a_chunk: np.ndarray, b_chunk: np.ndarray, p: int, d: int) -> tuple[float, int]:
    """Summed optimally-transported ground cost (+ #pairs) for ONE equal-size chunk.

    ``C[i, j] = ||a_i - b_j||_p^p / D`` (multidim MAE/MSE per pair); the exact
    ``linear_sum_assignment`` is the OT plan for equal-size uniform clouds. Takes
    numpy arrays so it is picklable for a ``ProcessPoolExecutor`` worker, and runs the
    SAME float64 ``torch.cdist`` math as the in-process path, so serial and parallel
    reduces agree bit-for-bit (modulo float summation order in the outer sum).
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    aa = torch.from_numpy(np.ascontiguousarray(a_chunk))
    bb = torch.from_numpy(np.ascontiguousarray(b_chunk))
    dist = torch.cdist(aa, bb, p=float(p))  # L1 (p=1) or L2 (p=2)
    cost = (dist.pow(2) if p == 2 else dist) / d
    row, col = linear_sum_assignment(cost.numpy())
    return float(cost[row, col].sum().item()), int(len(row))


def pooled_wasserstein_nd(
    pred, gt, *, p: int = 2, max_samples: int | None = None, seed: int = 0,
    chunked: bool = True, n_jobs: int = 1,
) -> float:
    """Multivariate ``p``-Wasserstein between two equal-size empirical action
    clouds via **exact optimal assignment**, with the full-vector ground cost.

    Unlike :func:`pooled_wasserstein_1d` (which compares the ``D`` marginals
    independently and is blind to cross-dimension structure), this treats each
    step's action as one point in ``R^D`` and transports the whole vectors. The
    ground cost is the **multidim MSE / MAE** between two D-vectors,
    ``c(x, y) = ||x - y||_p^p / D`` (``p=2`` ↔ multidim MSE, ``p=1`` ↔ multidim
    MAE). For equal-size, uniform-weight empirical clouds the optimal transport
    plan is a permutation, so ``W`` is the exact linear-sum assignment:

        ``W_p = [ (1/N) min_σ Σ_i c(pred_i, gt_{σ(i)}) ]^{1/p}``  (one scalar).

    Returns the actual ``p``-Wasserstein (the ``^{1/p}`` is applied): ``p=1`` →
    ``W1`` (mean L1 transported distance, MAE-scaled); ``p=2`` → ``W2`` (root of
    the mean squared transported distance, RMSE-scaled — *not* ``W2²``).
    ``W = 0`` iff the two clouds are identical multisets; the per-step pairing
    ``i↔i`` is dropped (forcing ``σ = identity`` recovers the plain ``mse`` /
    ``mae``).

    Cost is ``O(N^2)`` memory + ``~O(N^3)`` time, so for ``N > max_samples`` we cap
    the assignment size. Two capping modes:

    * ``chunked=True`` (default) — **divide-and-conquer / averaged minibatch OT**:
      seeded-shuffle both clouds, partition into ``ceil(N / max_samples)`` chunks of
      ``max_samples`` points, solve the exact assignment per chunk, and return the
      ``p``-th root of the **point-weighted mean transported cost over ALL chunks**.
      Uses every point at linear ``O(N · max_samples^2)`` cost; reduces exactly to a
      single assignment when ``N <= max_samples``. A variance-reduced minibatch-OT
      estimate that upper-bounds the true joint W (transport restricted within
      chunks); the bias is set by ``max_samples``, not by ``N``.
    * ``chunked=False`` — legacy single seeded subsample to ``max_samples`` points.

    ``max_samples=None`` → exact assignment over the full cloud. Seeded (``seed``),
    so deterministic / reproducible.

    ``n_jobs`` (chunked path) distributes the independent per-chunk assignments across
    that many CPU worker processes. The chunk set is fixed by the seeded shuffle, so the
    value is identical to ``n_jobs=1`` (the serial default) for any worker count.
    """
    if p not in (1, 2):
        raise ValueError(f"pooled_wasserstein_nd: p must be 1 or 2, got {p}")
    # float64: the outer p-th root amplifies cdist's float32 self-distance
    # round-off (catastrophic cancellation in the matmul Euclidean formula), so
    # identical clouds wouldn't read as exactly 0 in float32.
    a = _to_torch(pred).to(torch.float64)
    b = _to_torch(gt).to(torch.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            "pooled_wasserstein_nd: pred/gt must be equal-size (N, D); "
            f"got {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    n, d = a.shape
    a_np, b_np = a.numpy(), b.numpy()

    # Exact over the full cloud (no cap, or already small enough).
    if max_samples is None or n <= max_samples:
        total, cnt = _assignment_cost(a_np, b_np, p, d)
        return float((total / cnt) ** (1.0 / p)) if cnt else 0.0

    g = torch.Generator().manual_seed(int(seed))
    if not chunked:
        # Legacy: one seeded minibatch (independent perms per cloud).
        ia = torch.randperm(n, generator=g)[:max_samples].numpy()
        ib = torch.randperm(n, generator=g)[:max_samples].numpy()
        total, cnt = _assignment_cost(a_np[ia], b_np[ib], p, d)
        return float((total / cnt) ** (1.0 / p)) if cnt else 0.0

    # Divide-and-conquer: seeded-shuffle (independent perms per cloud), then exact OT
    # per max_samples-sized chunk; point-weighted mean transported cost over every
    # chunk (so ALL points contribute), rooted once. W_p = (Σ cost / Σ pairs)^(1/p).
    #
    # The chunk set is fixed HERE by one seeded shuffle over the full N, independent of
    # the worker count, so distributing it across ``n_jobs`` CPU processes is exact:
    # each chunk's assignment is independent and the reduce is an order-independent sum.
    # ``n_jobs=1`` is the serial default (byte-identical to the pre-parallel code).
    # ``linear_sum_assignment`` is CPU-only, so this is CPU — not GPU — parallelism.
    # See tests/groot/core/training/open_loop/dashboard/multigpu_accelerate.md.
    perm_a = torch.randperm(n, generator=g).numpy()
    perm_b = torch.randperm(n, generator=g).numpy()
    chunks = [
        (perm_a[lo:lo + max_samples], perm_b[lo:lo + max_samples])
        for lo in range(0, n, max_samples)
    ]

    if n_jobs and int(n_jobs) > 1 and len(chunks) > 1:
        try:
            import multiprocessing as _mp  # noqa: PLC0415
            from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

            # 'spawn', NOT the Linux-default 'fork': the eval parent has CUDA initialized
            # (the policy sits on the GPU), and forking a CUDA-initialized process then
            # importing/using torch in the child DEADLOCKS. Spawned workers start clean and
            # run CPU-only cdist, so they never touch the inherited (broken) CUDA context.
            with ProcessPoolExecutor(max_workers=int(n_jobs), mp_context=_mp.get_context("spawn")) as ex:
                # Submit and reduce in chunk order so the sum is FP-deterministic and
                # matches the serial accumulation exactly.
                futures = [
                    ex.submit(_assignment_cost, a_np[ia], b_np[ib], p, d) for ia, ib in chunks
                ]
                results = [f.result() for f in futures]
            total = sum(t for t, _ in results)
            cnt = sum(c for _, c in results)
            return float((total / cnt) ** (1.0 / p)) if cnt else 0.0
        except Exception:  # noqa: BLE001
            logger.exception(
                "pooled_wasserstein_nd: parallel reduce (n_jobs=%s) failed; using serial", n_jobs
            )

    total, cnt = 0.0, 0
    for ia, ib in chunks:
        t, c = _assignment_cost(a_np[ia], b_np[ib], p, d)
        total += t
        cnt += c
    return float((total / cnt) ** (1.0 / p)) if cnt else 0.0


def pooled_wasserstein_report(
    pred, gt, *, max_samples: int | None = None, seed: int = 0, n_jobs: int = 1
) -> dict[str, float]:
    """All pooled Wasserstein variants on the same ``(N, D)`` clouds as one dict.

    Two families × two orders:

    * ``wasserstein_{1,2}_perdim`` — per-dimension marginal ``W_p`` averaged over
      dims (cheap; blind to cross-dim coupling); see :func:`pooled_wasserstein_1d`.
    * ``wasserstein_{1,2}_joint``  — multivariate exact-assignment ``W_p`` over the
      full action vectors (the principled joint W₁ / W₂; captures the
      joint distribution); see :func:`pooled_wasserstein_nd`. ``max_samples`` caps
      the assignment size for tractability (``None`` → exact).
    """
    return {
        "wasserstein_1_perdim": float(pooled_wasserstein_1d(pred, gt, p=1).mean().item()),
        "wasserstein_2_perdim": float(pooled_wasserstein_1d(pred, gt, p=2).mean().item()),
        "wasserstein_1_joint": pooled_wasserstein_nd(pred, gt, p=1, max_samples=max_samples, seed=seed, n_jobs=n_jobs),
        "wasserstein_2_joint": pooled_wasserstein_nd(pred, gt, p=2, max_samples=max_samples, seed=seed, n_jobs=n_jobs),
    }


def reverse_kl_from_samples(
    pred_samples,
    targets,
) -> torch.Tensor:
    """Reverse-KL Monte-Carlo estimate from ``M`` stochastic policy samples.

    Implements the estimator

        ``D̂_KL(q || p) ≈ (1 / M) Σ_m [ log q(â_m) - log p(â_m) ]``

    where the policy distribution is the unit-variance Gaussian mixture
    induced by the ``M`` samples,

        ``q(a) = (1 / M) Σ_k  N(a ;  â_k,  I)``,

    the data / ground-truth distribution is a unit-variance Gaussian
    centred at the target,

        ``p(a) = N(a ;  a*,  I)``,

    and the action sequence ``(T, D)`` is flattened to a single ``TD``-dim
    vector (independence across time and per-dim, consistent with both
    Gaussians having identity covariance). The expectation
    ``E_{a ~ q}[log q(a) - log p(a)]`` is approximated by averaging over
    the same ``M`` policy samples. ``log q(â_m)`` is computed with all
    ``M`` mixture components — *no* leave-one-out — matching the
    closed-form Gaussian-mixture density at ``â_m``.

    Args:
        pred_samples: ``(M, B, T, D)`` — ``M >= 2`` independent samples
            from the policy at each of ``B`` items.
        targets: ``(B, T, D)`` — ground-truth action sequences (treated
            as the mean of the reference Gaussian, identity covariance).

    Returns:
        Scalar ``torch.Tensor`` — mean reverse-KL across the batch.

    Notes:
        ``M = 1`` is degenerate (``q`` collapses to a single delta and
        ``log q(â_1) = log N(â_1; â_1, I) = const``, so the estimator
        becomes a constant ``-log p`` term plus a constant). The
        function tolerates ``M = 1`` for shape compatibility, but the
        result is not informative.

        This is a *bandwidth = 1* KDE; absolute values scale with
        ``TD`` (the dimensionality of the flattened action sequence), so
        it is comparable across training steps for a fixed model /
        action horizon, but not directly comparable across models with
        different action shapes.
    """
    s = _to_torch(pred_samples).to(torch.float32)
    mu = _to_torch(targets).to(torch.float32)

    if s.ndim != 4:
        raise ValueError(f"reverse_kl_from_samples: pred_samples must be (M, B, T, D); got {tuple(s.shape)}")
    if mu.ndim != 3:
        raise ValueError(f"reverse_kl_from_samples: targets must be (B, T, D); got {tuple(mu.shape)}")
    M, B, T, D = s.shape
    if mu.shape != (B, T, D):
        raise ValueError(
            f"reverse_kl_from_samples: shape mismatch — "
            f"pred_samples=(M={M}, B={B}, T={T}, D={D}), targets={tuple(mu.shape)}"
        )

    TD = T * D
    const = -0.5 * TD * math.log(2.0 * math.pi)

    A = s.permute(1, 0, 2, 3).reshape(B, M, TD)
    MU = mu.reshape(B, 1, TD)

    d2 = torch.cdist(A, A).pow(2)
    log_q_each = torch.logsumexp(const - 0.5 * d2, dim=-1) - math.log(M)

    d2p = ((A - MU) ** 2).sum(dim=-1)
    log_p_each = const - 0.5 * d2p

    rkl_each = (log_q_each - log_p_each).mean(dim=-1)
    return rkl_each.mean()


def sliced_wasserstein(
    pred,
    gt,
    *,
    n_proj: int = 128,
    p: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """Sliced ``p``-Wasserstein distance between two embedding clouds.

    A true metric on distributions (symmetric, triangle inequality) that —
    unlike a Gaussian-fit Fréchet/FVD — makes no parametric assumption, so it
    sees multimodality. Each cloud is a set of L2 feature vectors (e.g. DINOv3
    per-video embeddings). We project both clouds onto ``n_proj`` random unit
    directions, compute the closed-form 1-D ``p``-Wasserstein per direction via
    quantile matching (the 1-D optimal coupling), average the ``W_p^p`` over
    directions, and take the ``1/p`` root.

    Args:
        pred: ``(N, D)`` predicted-video embeddings.
        gt:   ``(M, D)`` ground-truth-video embeddings (``M`` may differ from ``N``).
        n_proj: number of random 1-D projections (Monte-Carlo slices).
        p: Wasserstein order (``p >= 1``).
        seed: RNG seed for the projection directions — fixed so the metric is
            reproducible across checkpoints (and so it satisfies the triangle
            inequality across calls, which share directions).

    Returns:
        Scalar ``torch.Tensor`` (fp32). ``0`` iff the two empirical
        distributions coincide.
    """
    a = _to_torch(pred).to(torch.float32)
    b = _to_torch(gt).to(torch.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"sliced_wasserstein: pred/gt must be (N, D) / (M, D); got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"sliced_wasserstein: embedding dims must match; got D={a.shape[1]} vs D={b.shape[1]}")

    D = a.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.randn(D, n_proj, generator=gen, dtype=torch.float32)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
    theta = theta.to(a.device)

    ap = a @ theta  # (N, n_proj)
    bp = b @ theta  # (M, n_proj)

    n_q = max(a.shape[0], b.shape[0])
    u = torch.linspace(0.0, 1.0, n_q, device=a.device, dtype=torch.float32)
    qa = torch.quantile(ap, u, dim=0)  # (n_q, n_proj)
    qb = torch.quantile(bp, u, dim=0)  # (n_q, n_proj)

    wp_p = (qa - qb).abs().pow(p).mean(dim=0)  # (n_proj,) W_p^p per slice
    sw_p = wp_p.mean()  # average W_p^p over slices
    return sw_p.clamp_min(0.0).pow(1.0 / p)


def energy_distance(pred, gt, *, squared: bool = False) -> torch.Tensor:
    """Energy distance between two embedding clouds (first-power L2 ground).

    ``ED(P, Q) = sqrt( 2 E||X - Y|| - E||X - X'|| - E||Y - Y'|| )`` with
    ``X, X' ~ P`` and ``Y, Y' ~ Q`` (V-statistic estimator). A true metric on
    distributions — it is the MMD with the negative-type energy kernel
    ``-||.||``: symmetric, ``0`` iff ``P == Q``, triangle inequality. It is
    parameter-free and sees all moments, so — unlike a Gaussian-fit
    Fréchet/FVD — it is not blind to multimodality.

    Uses the **first-power** L2 ground distance (``cdist`` default), NOT
    squared: squared-L2 collapses the estimator to ``||mean_P - mean_Q||^2``
    and stops seeing spread.

    Args:
        pred: ``(N, D)`` predicted-video embeddings.
        gt:   ``(M, D)`` ground-truth-video embeddings.
        squared: if ``True`` return ``ED^2`` (a semimetric) instead of ``ED``.

    Returns:
        Scalar ``torch.Tensor`` (fp32), ``>= 0``.
    """
    a = _to_torch(pred).to(torch.float32)
    b = _to_torch(gt).to(torch.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"energy_distance: pred/gt must be (N, D) / (M, D); got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"energy_distance: embedding dims must match; got D={a.shape[1]} vs D={b.shape[1]}")

    cross = torch.cdist(a, b).mean()
    within_p = torch.cdist(a, a).mean()
    within_g = torch.cdist(b, b).mean()
    ed_sq = (2.0 * cross - within_p - within_g).clamp_min(0.0)
    return ed_sq if squared else torch.sqrt(ed_sq)


def energy_score(pred_samples, gt, *, return_components: bool = False):
    """Energy score of ``M`` predicted-future embeddings against one GT.

    ``ES(P, y) = (1/M) Σ_i ||x_i - y|| - 1/(2 M (M-1)) Σ_{i != j} ||x_i - x_j||``

    The strictly *proper scoring rule* for a multimodal forecast evaluated
    against a single realised outcome (multivariate CRPS). In expectation per
    state it equals ``½ · ED^2(P, δ_y) >= 0``; averaged over many states it is
    minimised iff ``P`` equals the true conditional distribution. It decomposes
    as ``ES = accuracy - spread`` where ``accuracy = (1/M) Σ ||x_i - y||`` and
    ``spread = ½ · mean_{i != j} ||x_i - x_j||`` — so it rewards *calibrated*
    diversity, penalising both mode collapse and over-dispersion. It is the
    proper, parameter-free successor to :func:`reverse_kl_from_samples` (no
    unit-variance KDE) and the non-Gaussian, conditional counterpart of
    :func:`frechet_gaussian_over_time`.

    Note: the unbiased pairwise term means an *individual* state's score may be
    marginally negative on small ``M``; the cross-state aggregate (and, when
    comparing two checkpoints on the same eval set, the *difference* of
    aggregates) is the meaningful quantity — there the model-independent
    constant cancels and ``Δ(energy score) = Δ(energy distance)``.

    Args:
        pred_samples: ``(M, D)`` — ``M >= 2`` embedding samples of the
            predicted future for one state.
        gt: ``(D,)`` or ``(1, D)`` — the single realised future embedding.
        return_components: if ``True`` return ``(es, accuracy, spread)``.

    Returns:
        Scalar ``torch.Tensor`` (the energy score), or the 3-tuple
        ``(es, accuracy, spread)`` if ``return_components``.
    """
    x = _to_torch(pred_samples).to(torch.float32)
    y = _to_torch(gt).to(torch.float32)
    if x.ndim != 2:
        raise ValueError(f"energy_score: pred_samples must be (M, D); got {tuple(x.shape)}")
    M = x.shape[0]
    if M < 2:
        raise ValueError(f"energy_score: requires M >= 2 samples; got M={M}")
    y = y.reshape(1, -1)
    if y.shape[1] != x.shape[1]:
        raise ValueError(f"energy_score: embedding dims must match; got pred D={x.shape[1]} vs gt D={y.shape[1]}")

    accuracy = torch.cdist(x, y).mean()  # (M, 1) -> (1/M) Σ ||x_i - y||
    pair = torch.cdist(x, x)  # (M, M), diagonal == 0
    pair_sum = pair.sum() - pair.diagonal().sum()  # exclude i == j
    spread = 0.5 * pair_sum / (M * (M - 1))
    es = accuracy - spread
    if return_components:
        return es, accuracy, spread
    return es


def video_distributional_report(
    pred_cloud,
    gt_cloud,
    *,
    n_proj: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Bundle the pooled video distributional metrics + their GT-GT floors.

    Given the pooled embedding clouds (one vector per video, pooled across the
    eval set), returns sliced-Wasserstein and energy distance, each alongside a
    **GT-GT floor**: the same metric between two random halves of the GT cloud.
    The floor is the irreducible finite-sample value (a model cannot beat it in
    expectation), so interpret ``metric - floor`` as the excess distributional
    mismatch.

    Args:
        pred_cloud: ``(N, D)`` pooled predicted-video embeddings.
        gt_cloud:   ``(M, D)`` pooled ground-truth-video embeddings.
        n_proj: projections for the sliced-Wasserstein term.
        seed: RNG seed for projections + the GT half-split (reproducible).

    Returns:
        ``dict[str, float]``. My approximate / kernel pair with GT-GT floors:
        ``sliced_wasserstein`` (+ ``_floor``), ``energy_distance`` (+ ``_floor``).
        The ported *exact* pooled Wasserstein on the same clouds:
        ``wasserstein_2_joint`` (+ ``_floor``) — exact Hungarian-OT W2, the value
        ``sliced_wasserstein`` approximates — plus ``wasserstein_1_joint`` and the
        axis-aligned ``wasserstein_{1,2}_perdim`` (diagnostic; weak on learned
        embedding axes). And ``fvd`` (+ ``_floor``) — the FID/FVD Fréchet distance
        (parametric Gaussian fit) on the same cloud; canonical FVD when the cloud is
        I3D clip features, appearance-Fréchet on the default per-frame DINOv3 features.
    """
    a = _to_torch(pred_cloud).to(torch.float32)
    g = _to_torch(gt_cloud).to(torch.float32)

    sw = sliced_wasserstein(a, g, n_proj=n_proj, seed=seed)
    ed = energy_distance(a, g)
    # Exact pooled Wasserstein on the SAME (equal-size) pred/gt clouds: the joint
    # W2 here is the exact value `sliced_wasserstein` approximates — both use the
    # per-dim-normalised (`/D`) ground cost, so they share a scale and read side
    # by side. Feasible because the per-segment video clouds are small (N=#videos).
    wreport = pooled_wasserstein_report(a, g, max_samples=None)

    n = g.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    h = n // 2
    g1 = g[perm[:h]]
    g2 = g[perm[h : 2 * h]]
    # FVD-style Fréchet distance (parametric Gaussian-fit Bures) on the SAME cloud —
    # canonical FVD when the cloud is I3D clip features; appearance-Fréchet on DINOv3.
    fvd = frechet_distance(a, g)
    sw_floor = sliced_wasserstein(g1, g2, n_proj=n_proj, seed=seed)
    ed_floor = energy_distance(g1, g2)
    wj2_floor = pooled_wasserstein_nd(g1, g2, p=2, max_samples=None)
    fvd_floor = frechet_distance(g1, g2)

    return {
        "sliced_wasserstein": float(sw),
        "sliced_wasserstein_floor": float(sw_floor),
        "energy_distance": float(ed),
        "energy_distance_floor": float(ed_floor),
        "wasserstein_2_joint": wreport["wasserstein_2_joint"],
        "wasserstein_2_joint_floor": float(wj2_floor),
        "wasserstein_1_joint": wreport["wasserstein_1_joint"],
        "wasserstein_2_perdim": wreport["wasserstein_2_perdim"],
        "wasserstein_1_perdim": wreport["wasserstein_1_perdim"],
        "fvd": float(fvd),
        "fvd_floor": float(fvd_floor),
    }
