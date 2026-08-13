# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Import-light construction boundary for the GrootN2 policy.

Two jobs, both of which must happen before anything heavy is imported:

1. **Authenticate the strict task config.** An invalid, stale or forged YAML has
   to fail before the checkpoint is inspected and before
   ``ensure_cosmos_model_deps`` can download or mutate a dependency cache.
2. **Choose the policy implementation.** ``Gr00tN2RLPolicy`` — the
   ``Gr00tN2Policy`` subclass under ``groot.core.rl`` that adds frozen-critic
   action reranking and the strict xDoF admission boundary — is selected by an
   authenticated task binding or an environment selector. Choosing it here —
   rather than branching inside the base policy — is what keeps
   ``groot.core.policy`` free of any RL import, and therefore what lets the
   curated OSS package ship without ``groot.core.rl`` at all. Because an
   environment variable alone can decide between the two classes, the one that
   won is also *named* here, at INFO: this is the only point in the process that
   knows which of them is about to serve.

Both RL entry points below are **optional by construction**: this module imports
neither torch nor transformers at module scope, and every ``groot.core.rl``
import is guarded. With the package absent — which is the state of the tree
until ``Gr00tN2RLPolicy`` lands, and the permanent state of the OSS package —
the factory behaves exactly like calling ``Gr00tN2Policy(...)`` directly.
"""

from __future__ import annotations

import logging
import os

_RL_PACKAGE = "groot.core.rl"
_RL_POLICY_MODULE = "groot.core.rl.rl100_n2.n2_rl_policy"

# The subclass by name, for the one log line that has to name a class it could
# not import. Duplicated for the same reason ``STRICT_TASK_CONFIG_ENV`` is, and
# pinned by ``test_the_rl_class_name_matches_the_rl_package`` wherever the real
# package IS importable.
_RL_POLICY_QUALNAME = f"{_RL_POLICY_MODULE}.Gr00tN2RLPolicy"

# Env selectors that only ``Gr00tN2RLPolicy`` reads. Kept here, in the
# construction boundary, because this is the one place that can act on them.
RL_ONLY_ENV_PREFIXES = ("IDQL_", "RL100_N2_")

# The environment source of the strict task config. Duplicated from
# ``rl100_n2.task_config.STRICT_POLICY_TASK_CONFIG_ENV`` on purpose: the
# branches that have to honor it are exactly the branches in which that module
# could not be imported. ``test_strict_env_name_matches_the_rl_package`` fails
# if the two ever drift.
#
# It matches ``RL_ONLY_ENV_PREFIXES`` too, but it is NOT an ordinary selector:
# it names a strict authority file, so it degrades like ``--task-config``
# (refuse) rather than like ``IDQL_K`` (warn).
STRICT_TASK_CONFIG_ENV = "RL100_N2_TASK_CONFIG"

_logger = logging.getLogger(__name__)


def rl_only_env_requests() -> list[str]:
    """Return the RL-only env selectors currently set to a non-blank value.

    Blank is not a request: an exported-but-empty variable is how a launcher
    clears a knob it does not want.
    """
    return sorted(
        name
        for name, value in os.environ.items()
        if name.startswith(RL_ONLY_ENV_PREFIXES) and value.strip()
    )


def strict_task_config_env_request() -> str | None:
    """Return the strict task config configured in the environment, if any.

    Same blank-is-not-a-request rule as :func:`rl_only_env_requests`.
    """
    return os.environ.get(STRICT_TASK_CONFIG_ENV, "").strip() or None


def configured_strict_task_config(explicit_path: str | os.PathLike | None = None) -> str | None:
    """Return the strict task config requested by either source, or ``None``.

    ONE definition of "configured", deliberately mirroring the resolution in
    ``authenticate_strict_task_config``: strip both sources, explicit wins, and
    blank means unconfigured. The missing-package branches below refuse exactly
    when this returns non-``None``, so they refuse exactly when the real
    authenticator would have produced a binding — no false refusal on a
    degenerate ``""``, and no silent pass on a configured environment.

    Conflict detection is NOT reproduced here: with the package absent, a
    conflicting pair and an agreeing pair are both refused anyway.
    """
    explicit = "" if explicit_path is None else str(explicit_path).strip()
    return explicit or strict_task_config_env_request()


def _qualified(policy_cls) -> str:
    """``module.QualName`` — the spelling that tells the two classes apart.

    ``Gr00tN2RLPolicy`` subclasses ``Gr00tN2Policy`` and its own startup prints
    carry the base class's name as their prefix, so the dotted path — which puts
    ``groot.core.rl`` in front of the reader — is what makes the answer
    unambiguous.
    """
    return f"{policy_cls.__module__}.{policy_cls.__qualname__}"


def _selection_source(binding, rl_requests) -> str:
    """Name whatever selected the subclass, for the serving log line."""
    sources = []
    if binding is not None:
        sources.append("an authenticated strict task config")
    if rl_requests:
        sources.append(f"environment selector(s) {list(rl_requests)}")
    return " and ".join(sources)


def _missing_rl_package(exc: ModuleNotFoundError) -> bool:
    """Whether ``exc`` is 'the RL package is absent' rather than a real bug.

    A broken import *inside* ``groot.core.rl`` (a typo, a missing third-party
    dependency) also raises ``ModuleNotFoundError``, and swallowing that would
    turn a bug into a silent fall back to base N2. Only an absent
    ``groot.core.rl`` — or an absent module directly under it — counts.
    """
    return (exc.name or "") in (_RL_PACKAGE, _RL_POLICY_MODULE)


def authenticate_server_task_config(explicit_path: str | None, *, logger=None):
    """Resolve + authenticate a strict task config at a server entrypoint.

    Explicit configuration and the ``RL100_N2_TASK_CONFIG`` environment override
    are resolved together (a conflicting pair fails closed). Returns the
    authenticated binding object, or ``None`` when neither source is configured
    — which is the exact legacy behavior.

    Call this FIRST, before the checkpoint is inspected and before any
    dependency preparation. The point is side-effect ordering, not import
    weight: pulling ``rl100_n2.task_config`` executes that package's
    ``__init__``, which is light only relative to the policy implementation,
    not absolutely.
    """
    try:
        from groot.core.rl.rl100_n2.task_config import authenticate_strict_task_config
    except ModuleNotFoundError as exc:
        if not _missing_rl_package(exc):
            raise
        # Check BOTH configured sources. Checking only ``explicit_path`` would
        # let ``RL100_N2_TASK_CONFIG`` fall through to the env-selector path,
        # which warns and serves base N2 — the silent degradation this branch
        # exists to prevent.
        configured = configured_strict_task_config(explicit_path)
        if configured:
            # A strict task config is a security-shaped contract. Quietly
            # serving without it would be worse than refusing to serve.
            raise RuntimeError(
                f"task_config was configured ({configured}), but this package "
                f"ships without {_RL_PACKAGE}, so no strict task authority can "
                "be established"
            ) from exc
        return None

    binding = authenticate_strict_task_config(explicit_path=explicit_path)
    if binding is not None and logger is not None:
        logger.info(
            "Authenticated strict task config %s (profile=%s, sha256=%s)",
            binding.source_path,
            binding.task_profile,
            binding.source_sha256,
        )
    return binding


def resolve_policy_class(*, binding=None, rl_requests=None):
    """Return ``(policy_class, is_rl)`` for the requested inference behavior.

    ``Gr00tN2RLPolicy`` is selected by an authenticated task binding or by an
    ``IDQL_*`` / ``RL100_N2_*`` environment selector. When nothing selects it —
    or when a plain selector does and the RL package is simply not installed —
    this returns the base ``Gr00tN2Policy`` and the caller gets ordinary N2
    inference.

    A strict task config is the one request that never degrades that way,
    whether it arrives as a binding or as ``RL100_N2_TASK_CONFIG``.

    **Every returning branch names the class it is about to serve, at INFO.**
    Nothing else in the process does. Upstream, the server prints ``Policy type:
    groot_n2``, which describes the *checkpoint* and is equally true of both
    classes. Downstream, ``Gr00tN2RLPolicy`` announces itself only when critic
    reranking is on, and under ``Gr00tN2Policy``'s name at that. So an
    environment selector can decide which of the two classes runs and leave no
    trace of it anywhere.

    The line is unconditional rather than only-when-the-subclass-wins on
    purpose: were it conditional, its absence would mean either "the base class
    served" or "this build predates the line", and a log that cannot separate
    those does not settle the question it exists to settle.
    """
    rl_requests = rl_only_env_requests() if rl_requests is None else rl_requests
    if binding is None and not rl_requests:
        from groot.core.policy.n2_policy import Gr00tN2Policy

        _logger.info(
            "Serving %s (no selector set).",
            _qualified(Gr00tN2Policy),
        )
        return Gr00tN2Policy, False

    selected_by = _selection_source(binding, rl_requests)
    try:
        from groot.core.rl.rl100_n2.n2_rl_policy import Gr00tN2RLPolicy
    except ModuleNotFoundError as exc:
        if not _missing_rl_package(exc):
            raise
        if binding is not None:
            # Unreachable in practice — a binding can only exist if the package
            # imported a moment ago — but a strict contract must never degrade
            # silently, so state the invariant rather than assume it.
            raise RuntimeError(
                "a strict task binding was authenticated, but the RL policy "
                f"implementation ({_RL_POLICY_MODULE}) is unavailable"
            ) from exc
        if STRICT_TASK_CONFIG_ENV in rl_requests:
            # Not an ordinary selector: it names a strict authority file, so it
            # obeys the --task-config contract (refuse) rather than the IDQL_*
            # one (warn). Normally ``authenticate_server_task_config`` has
            # already refused upstream; this covers a direct caller that skipped
            # that boundary.
            raise RuntimeError(
                f"{STRICT_TASK_CONFIG_ENV} configures a strict task config, but "
                f"this package ships without {_RL_PACKAGE}, so no strict task "
                "authority can be established"
            ) from exc
        # Env-only request with no RL package: serve base N2 rather than refuse,
        # but say so loudly. Silence here is how an eval run reports ordinary N2
        # numbers under a name that promises critic reranking.
        _logger.warning(
            "%s select %s, but this package ships without %s. "
            "Serving the base Gr00tN2Policy — these variables have NO effect.",
            rl_requests,
            _RL_POLICY_QUALNAME,
            _RL_PACKAGE,
        )
        from groot.core.policy.n2_policy import Gr00tN2Policy

        # Deliberately redundant with the warning above: the warning has to stay
        # self-contained for a process filtering at WARNING, and this line has to
        # appear on every branch for its absence to mean anything.
        _logger.info(
            "Serving %s (%s selected %s, but %s is not installed).",
            _qualified(Gr00tN2Policy),
            selected_by,
            _RL_POLICY_QUALNAME,
            _RL_PACKAGE,
        )
        return Gr00tN2Policy, False

    _logger.info(
        "Serving %s (selected by %s).",
        _qualified(Gr00tN2RLPolicy),
        selected_by,
    )
    return Gr00tN2RLPolicy, True


def create_gr00t_n2_policy(
    *,
    model_path: str,
    task_config_path: str | None = None,
    strict_task_binding=None,
    **policy_kwargs,
):
    """Create the N2 policy after the import-light authority boundary.

    ``strict_task_binding`` is used by server bootstraps that already resolved
    explicit/environment selection before checkpoint inspection. It is freshly
    compared with the same live source here, immediately before dependency
    preparation, and the resulting immutable object is passed to the heavy
    constructor without a later path read.
    """
    try:
        from groot.core.rl.rl100_n2.task_config import (
            STRICT_POLICY_TASK_CONFIG_ENV,
            authenticate_strict_task_config,
            verify_task_authority,
        )
    except ModuleNotFoundError as exc:
        if not _missing_rl_package(exc):
            raise
        # Same one definition of "configured" as the server bootstrap: the
        # environment counts, because a direct caller that passes neither
        # argument still gets the strict contract from it.
        if configured_strict_task_config(task_config_path) or strict_task_binding is not None:
            raise RuntimeError(
                "a strict N2 task config was requested, but this package ships "
                f"without {_RL_PACKAGE}, so no task authority can be established"
            ) from exc
        # Neither source configured is the exact legacy behavior of
        # ``authenticate_strict_task_config``: no binding.
        binding = None
    else:
        if strict_task_binding is None:
            binding = authenticate_strict_task_config(explicit_path=task_config_path)
        else:
            # Resolve configured explicit/environment selection again so a direct
            # caller cannot smuggle a pre-built binding past it. With neither
            # source configured, freshly verify the supplied immutable identity.
            # Do not feed its normalized absolute source back as an explicit
            # path: a legitimate relative environment spelling would then look
            # like a conflict despite naming the already-authenticated file.
            configured_env = os.environ.get(STRICT_POLICY_TASK_CONFIG_ENV, "").strip()
            if task_config_path or configured_env:
                binding = authenticate_strict_task_config(explicit_path=task_config_path)
            else:
                binding = verify_task_authority(strict_task_binding.authority())
            if binding is None or binding.receipt() != strict_task_binding.receipt():
                raise ValueError(
                    "pre-authenticated strict task binding does not match the live source bytes"
                )

    # Decide the implementation before any heavy import, so a request that
    # cannot be served fails here rather than after a checkpoint download.
    policy_cls, is_rl = resolve_policy_class(binding=binding)

    # Heavy runtime imports begin only after the final live-byte check above.
    # Auto-download Cosmos DiT / text encoder / Wan VAE and rewrite baked Lustre
    # paths in config.json + processor_config.json to point at the local cache.
    # No-op when the paths already exist locally.
    from groot.core.training.models.n2.utils.cosmos_model_deps import (
        ensure_cosmos_model_deps,
    )

    ensure_cosmos_model_deps(model_path)

    if is_rl:
        return policy_cls(model_path=model_path, strict_task_binding=binding, **policy_kwargs)
    # The base class has no strict-binding parameter — that contract belongs to
    # the RL subclass — and ``binding is None`` is guaranteed on this branch.
    return policy_cls(model_path=model_path, **policy_kwargs)
