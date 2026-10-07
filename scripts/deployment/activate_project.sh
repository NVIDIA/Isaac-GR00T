#!/usr/bin/env bash
# Shared project selection for platform activation scripts. Source, do not execute.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script: source scripts/deployment/activate_project.sh <platform>" >&2
    exit 2
fi

_groot_activate_project() {
    local deployment_dir project name
    deployment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    case "${1:-}" in
        dgpu) project="$(cd "$deployment_dir/../.." && pwd)" ;;
        thor) project="$deployment_dir/thor" ;;
        *) echo "Select a platform: dgpu or thor." >&2; return 2 ;;
    esac
    if [[ ! -f "$project/pyproject.toml" || ! -f "$project/uv.lock" ]]; then
        echo "ERROR: ${1} has no platform manifest and lockfile yet." >&2
        return 1
    fi
    if [[ ! -f "$project/.venv/bin/activate" ]]; then
        echo "ERROR: Environment missing at $project/.venv; run the platform installer first." >&2
        return 1
    fi
    if [[ "$1" != dgpu && "$(realpath "$project/.venv")" == "$(realpath -m "$deployment_dir/../../.venv")" ]]; then
        echo "ERROR: $1 must use a separate environment from the root project's .venv." >&2
        return 1
    fi

    # Restore the previous selection before switching platforms or reactivating.
    if declare -F _groot_venv_deactivate >/dev/null; then
        deactivate
    fi
    # shellcheck disable=SC1091
    source "$project/.venv/bin/activate" || return
    if ! command -v uv >/dev/null 2>&1 && [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is missing; run the platform installer first." >&2
        deactivate
        return 1
    fi
    _GROOT_SAVED_UV_NAMES=(UV_PROJECT UV_PROJECT_ENVIRONMENT UV_LOCKED UV_NO_SYNC)
    _GROOT_SAVED_UV_VALUES=()
    _GROOT_SAVED_UV_SET=()
    for name in "${_GROOT_SAVED_UV_NAMES[@]}"; do
        _GROOT_SAVED_UV_VALUES+=("${!name-}")
        if [[ -v "$name" ]]; then
            _GROOT_SAVED_UV_SET+=(1)
        else
            _GROOT_SAVED_UV_SET+=(0)
        fi
    done

    # Wrap the venv's own deactivate so standard shell deactivation also restores uv.
    eval "$(declare -f deactivate | sed '1s/deactivate/_groot_venv_deactivate/')"
    deactivate() {
        local name index
        _groot_venv_deactivate "$@"
        for index in "${!_GROOT_SAVED_UV_NAMES[@]}"; do
            name="${_GROOT_SAVED_UV_NAMES[$index]}"
            if [[ "${_GROOT_SAVED_UV_SET[$index]}" == 1 ]]; then
                export "$name=${_GROOT_SAVED_UV_VALUES[$index]}"
            else
                unset "$name"
            fi
        done
        unset _GROOT_SAVED_UV_NAMES _GROOT_SAVED_UV_VALUES _GROOT_SAVED_UV_SET
        unset -f _groot_venv_deactivate deactivate
    }
    export UV_PROJECT="$project"
    # Keep this relative: losing UV_PROJECT must not point the root project at Thor's env.
    export UV_PROJECT_ENVIRONMENT=.venv
    export UV_LOCKED=true
    unset UV_NO_SYNC
    echo "uv project: $UV_PROJECT"
    echo "uv environment: $project/.venv"
}

_groot_activate_project "$@"
_groot_activation_status=$?
unset -f _groot_activate_project
return "$_groot_activation_status"
