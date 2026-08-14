"""Utilities for converting objects back into tyro-compatible CLI args."""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import enum
import os
import re
import types
import typing
from typing import Union, get_args, get_origin, overload

_MISSING = object()
T = typing.TypeVar("T")


class DataclassInstance(typing.Protocol):
    """Structural type for dataclass instances used by overlay helpers."""

    __dataclass_fields__: typing.ClassVar[dict[str, object]]


OverrideValues = collections.abc.Mapping[object, object] | DataclassInstance
_OVERLAY_TYPE_CACHE: dict[type[object], type[DataclassInstance]] = {}


class UnsetType:
    """Sentinel type for fields that should be skipped by ``apply_overrides``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: typing.Final[UnsetType] = UnsetType()


class UntyroUnsupportedOverrideError(ValueError):
    """Raised when a changed value cannot be represented as tyro argv."""


def _field_to_name(name: str, prefix: str) -> str:
    """Convert a dataclass field name and optional prefix to a tyro CLI name."""
    full = f"{prefix}.{name}" if prefix else name
    parts = full.split(".")
    return ".".join(p.replace("_", "-") for p in parts)


def _field_to_flag(name: str, prefix: str) -> str:
    """Convert a dataclass field name and optional prefix to a tyro CLI flag."""
    return "--" + _field_to_name(name, prefix)


def _is_union(tp: object) -> bool:
    origin = get_origin(tp)
    return origin is Union or origin is types.UnionType


def _is_annotated(tp: object) -> bool:
    return get_origin(tp) is typing.Annotated


def _strip_annotated(tp: object) -> tuple[object, tuple[object, ...]]:
    if not _is_annotated(tp):
        return tp, ()
    base, *metadata = get_args(tp)
    return base, tuple(metadata)


def _tyro_default_selector_name(tp: object) -> str | None:
    name = getattr(tp, "__name__", None)
    if not isinstance(name, str):
        return None
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    return name.replace("_", "-").lower()


def _is_structured_type(tp: object) -> bool:
    return (isinstance(tp, type) and dataclasses.is_dataclass(tp)) or typing.is_typeddict(
        tp
    )


def _is_optional(tp: object) -> bool:
    if _is_union(tp):
        return type(None) in get_args(tp)
    return False


def _unwrap_optional(tp: type) -> object:
    return next(a for a in get_args(tp) if a is not type(None))


def _is_bool(tp: type) -> bool:
    return tp is bool


def _is_optional_bool(tp: type) -> bool:
    return _is_optional(tp) and _is_bool(_unwrap_optional(tp))


def _negate_flag(flag: str) -> str:
    """Return tyro's negated form for a boolean flag."""
    body = flag[2:]
    if "." in body:
        prefix, last = body.rsplit(".", 1)
        return f"--{prefix}.no-{last}"
    return f"--no-{body}"


def _scalar_to_str(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.name
    return str(value)


def _extend_sequence_tokens(
    tokens: list[str], values: list[object] | tuple[object, ...]
) -> None:
    for item in values:
        if isinstance(item, (list, tuple)):
            _extend_sequence_tokens(tokens, item)
        else:
            tokens.append(_scalar_to_str(item))


def _is_cli_representable_sequence(value: object) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    for item in value:
        if isinstance(item, (list, tuple)):
            if not _is_cli_representable_sequence(item):
                return False
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            return False
        elif isinstance(item, dict):
            return False
    return True


def _resolve_type_hints(instance: object) -> dict[str, type]:
    cls = type(instance)
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return {f.name: f.type for f in dataclasses.fields(instance)}


def _subcommand_name_from_metadata(metadata: tuple[object, ...]) -> str | None:
    for item in metadata:
        name = getattr(item, "name", None)
        if isinstance(name, str):
            return name
    return None


def _subcommand_choice_for_value(
    value: object, field_type: object
) -> tuple[str, object] | None:
    field_type, _ = _strip_annotated(field_type)
    if not _is_union(field_type):
        return None

    for choice_type in get_args(field_type):
        choice_type, metadata = _strip_annotated(choice_type)
        command_name = _subcommand_name_from_metadata(metadata)
        if command_name is None or not isinstance(choice_type, type):
            continue
        if isinstance(value, choice_type):
            return command_name, choice_type
    return None


def _optional_structured_choice(field_type: object) -> tuple[str, object] | None:
    if not _is_optional(field_type):
        return None

    choice_type, metadata = _strip_annotated(_unwrap_optional(field_type))
    if not _is_structured_type(choice_type):
        return None

    command_name = _subcommand_name_from_metadata(metadata)
    if command_name is None:
        command_name = _tyro_default_selector_name(choice_type)
    if command_name is None:
        return None
    return command_name, choice_type


def _default_for_structured_type(tp: object) -> object:
    if typing.is_typeddict(tp):
        return {}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        try:
            return tp()
        except TypeError:
            return _MISSING
    return _MISSING


def _values_equal(left: object, right: object) -> bool:
    if right is _MISSING:
        return False
    try:
        return left == right
    except Exception:
        return False


def _raise_unsupported(path: str, value: object) -> None:
    raise UntyroUnsupportedOverrideError(
        f"Changed value at {path!r} cannot be represented as tyro argv: "
        f"{type(value).__name__}"
    )


def _serialize_field(value: object, field_type: object, flag: str) -> list[str]:
    if value is None:
        if _optional_structured_choice(field_type) is not None:
            return [f"{flag[2:]}:None"]
        if _is_optional(field_type):
            return [flag, "None"]
        return []

    if _is_optional_bool(field_type):
        return [flag, str(value)]

    if _is_bool(field_type):
        if value:
            return [flag]
        return [_negate_flag(flag)]

    subcommand_choice = _subcommand_choice_for_value(value, field_type)
    if subcommand_choice is not None:
        command_name, _ = subcommand_choice
        arg_name = flag[2:]
        return [f"{arg_name}:{command_name}"] + _dataclass_to_args(value, prefix=arg_name)

    optional_structured_choice = _optional_structured_choice(field_type)
    if optional_structured_choice is not None:
        command_name, _ = optional_structured_choice
        arg_name = flag[2:]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [f"{arg_name}:{command_name}"] + _dataclass_to_args(value, prefix=arg_name)
        if isinstance(value, dict):
            return [f"{arg_name}:{command_name}"] + _dict_to_args(value, prefix=arg_name)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_args(value, prefix=flag[2:])

    if isinstance(value, dict):
        return _dict_to_args(value, prefix=flag[2:])

    if isinstance(value, enum.Enum):
        return [flag, value.name]

    if isinstance(value, (list, tuple)):
        tokens = [flag]
        _extend_sequence_tokens(tokens, value)
        return tokens

    return [flag, _scalar_to_str(value)]


def _dataclass_to_args(instance: object, prefix: str = "") -> list[str]:
    """Convert a dataclass instance to tyro CLI argument strings."""
    if not dataclasses.is_dataclass(instance) or isinstance(instance, type):
        raise TypeError(f"Expected a dataclass instance, got {type(instance)}")

    resolved_hints = _resolve_type_hints(instance)
    args: list[str] = []
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        flag = _field_to_flag(field.name, prefix)
        field_type = resolved_hints.get(field.name, field.type)
        args.extend(_serialize_field(value, field_type, flag))
    return args


def _dict_to_args(d: dict[str, object], prefix: str = "") -> list[str]:
    """Convert a dict of named values to tyro CLI argument strings."""
    args: list[str] = []
    for key, value in d.items():
        flag = _field_to_flag(key, prefix)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            args.extend(_dataclass_to_args(value, prefix=flag[2:]))
        elif isinstance(value, dict):
            args.extend(_dict_to_args(value, prefix=flag[2:]))
        elif isinstance(value, bool):
            args.append(flag if value else _negate_flag(flag))
        elif value is None:
            args.extend([flag, "None"])
        elif isinstance(value, enum.Enum):
            args.extend([flag, value.name])
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            _extend_sequence_tokens(args, value)
        else:
            args.extend([flag, _scalar_to_str(value)])
    return args


def _serialize_override_value(
    value: object,
    default: object,
    field_type: object,
    flag: str,
) -> list[str]:
    if _values_equal(value, default):
        return []

    arg_name = flag[2:]
    subcommand_choice = _subcommand_choice_for_value(value, field_type)
    if subcommand_choice is not None:
        command_name, choice_type = subcommand_choice
        default_choice = (
            _subcommand_choice_for_value(default, field_type)
            if default is not _MISSING
            else None
        )
        branch_changed = (
            default_choice is None or default_choice[0] != command_name
        )
        args = [f"{arg_name}:{command_name}"]
        branch_default = default
        if branch_changed:
            branch_default = _default_for_structured_type(choice_type)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            args.extend(_dataclass_overrides(value, branch_default, prefix=arg_name))
            return args
        _raise_unsupported(arg_name, value)

    optional_structured_choice = _optional_structured_choice(field_type)
    if optional_structured_choice is not None:
        command_name, choice_type = optional_structured_choice
        if value is None:
            return [f"{arg_name}:None"]

        selector_needed = default is _MISSING or default is None
        args = [f"{arg_name}:{command_name}"]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            nested_default = (
                _default_for_structured_type(choice_type)
                if selector_needed
                else default
            )
            args.extend(_dataclass_overrides(value, nested_default, prefix=arg_name))
            return args
        if isinstance(value, dict):
            nested_default = {} if selector_needed else default
            if not isinstance(nested_default, dict):
                _raise_unsupported(arg_name, value)
            args.extend(_dict_overrides(value, nested_default, prefix=arg_name))
            return args
        _raise_unsupported(arg_name, value)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if default is _MISSING:
            nested_default = _default_for_structured_type(type(value))
        else:
            nested_default = default
        args = _dataclass_overrides(value, nested_default, prefix=arg_name)
        if args:
            return args
        if default is _MISSING:
            _raise_unsupported(arg_name, value)
        return []

    if isinstance(value, dict):
        nested_default = default if isinstance(default, dict) else {}
        if default not in (_MISSING, None) and not isinstance(default, dict):
            _raise_unsupported(arg_name, value)
        return _dict_overrides(value, nested_default, prefix=arg_name)

    if isinstance(value, (list, tuple)):
        if not _is_cli_representable_sequence(value):
            _raise_unsupported(arg_name, value)
        return _serialize_field(value, field_type, flag)

    if isinstance(value, bool) and not _is_optional_bool(field_type):
        return [flag] if value else [_negate_flag(flag)]

    return _serialize_field(value, field_type, flag)


def _dataclass_overrides(
    instance: object,
    default: object = _MISSING,
    prefix: str = "",
    field_order: tuple[str, ...] | None = None,
) -> list[str]:
    if not dataclasses.is_dataclass(instance) or isinstance(instance, type):
        raise TypeError(f"Expected a dataclass instance, got {type(instance)}")

    if default is not _MISSING and (
        not dataclasses.is_dataclass(default) or isinstance(default, type)
    ):
        _raise_unsupported(prefix or type(instance).__name__, instance)

    resolved_hints = _resolve_type_hints(instance)
    fields = list(dataclasses.fields(instance))
    if field_order is not None:
        order = {name: index for index, name in enumerate(field_order)}
        fields.sort(key=lambda field: order.get(field.name, len(order)))

    args: list[str] = []
    for field in fields:
        value = getattr(instance, field.name)
        default_value = (
            getattr(default, field.name)
            if default is not _MISSING and hasattr(default, field.name)
            else _MISSING
        )
        flag = _field_to_flag(field.name, prefix)
        field_type = resolved_hints.get(field.name, field.type)
        args.extend(_serialize_override_value(value, default_value, field_type, flag))
    return args


def _dict_overrides(
    values: dict[object, object],
    defaults: dict[object, object] | object = _MISSING,
    prefix: str = "",
) -> list[str]:
    if defaults is _MISSING:
        defaults = {}
    if not isinstance(defaults, dict):
        _raise_unsupported(prefix or "dict", values)

    removed_keys = set(defaults) - set(values)
    if removed_keys:
        removed = ", ".join(str(key) for key in sorted(removed_keys, key=str))
        raise UntyroUnsupportedOverrideError(
            f"Changed dict at {prefix!r} removes keys that tyro argv cannot remove: {removed}"
        )

    args: list[str] = []
    for key, value in values.items():
        default_value = defaults.get(key, _MISSING)
        flag = _field_to_flag(str(key), prefix)
        args.extend(_serialize_override_value(value, default_value, object, flag))
    return args


def _override_items(values: OverrideValues) -> tuple[tuple[str, object], ...]:
    if isinstance(values, collections.abc.Mapping):
        return tuple((str(key), value) for key, value in values.items())
    if dataclasses.is_dataclass(values) and not isinstance(values, type):
        return tuple(
            (field.name, getattr(values, field.name))
            for field in dataclasses.fields(values)
        )
    raise TypeError(
        "Expected override values to be a dict or dataclass instance, "
        f"got {type(values)}"
    )


def _apply_override_values(
    target: object, values: OverrideValues, path: str = ""
) -> None:
    values_are_dataclass = dataclasses.is_dataclass(values) and not isinstance(
        values, type
    )

    for key, value in _override_items(values):
        if isinstance(value, UnsetType):
            continue

        item_path = f"{path}.{key}" if path else key
        should_merge_value = isinstance(value, collections.abc.Mapping) or (
            values_are_dataclass
            and dataclasses.is_dataclass(value)
            and not isinstance(value, type)
        )

        if isinstance(target, dict):
            current = target.get(key, _MISSING)
            if (
                should_merge_value
                and (
                    isinstance(current, dict)
                    or (
                        dataclasses.is_dataclass(current)
                        and not isinstance(current, type)
                    )
                )
            ):
                _apply_override_values(
                    current, typing.cast(OverrideValues, value), item_path
                )
            else:
                target[key] = value
            continue

        if dataclasses.is_dataclass(target) and not isinstance(target, type):
            field_names = {field.name for field in dataclasses.fields(target)}
            if key not in field_names:
                raise UntyroUnsupportedOverrideError(
                    f"Override path {item_path!r} does not exist on {type(target).__name__}"
                )

            current = getattr(target, key)
            if (
                should_merge_value
                and (
                    isinstance(current, dict)
                    or (
                        dataclasses.is_dataclass(current)
                        and not isinstance(current, type)
                    )
                )
            ):
                _apply_override_values(
                    current, typing.cast(OverrideValues, value), item_path
                )
            else:
                setattr(target, key, value)
            continue

        _raise_unsupported(path or type(target).__name__, target)


def apply_overrides(default: T, values: OverrideValues) -> T:
    """Return a deep copy of ``default`` with nested override values applied.

    ``values`` may be a nested dict or a dataclass instance with fields matching
    a subset of the target dataclass. Fields set to ``UNSET`` are skipped, while
    ``None`` is treated as an explicit override value.
    """
    target = copy.deepcopy(default)
    _apply_override_values(target, values)
    return target


def _overlay_annotation(field_type: object) -> object:
    try:
        return field_type | UnsetType  # type: ignore[operator]
    except TypeError:
        try:
            return typing.Union[field_type, UnsetType]  # type: ignore[valid-type]
        except Exception:
            return object
    except Exception:
        return object


def overlay_type(dataclass_type: type[object]) -> type[DataclassInstance]:
    """Return a cached sparse overlay dataclass for ``dataclass_type``."""
    if not dataclasses.is_dataclass(dataclass_type) or not isinstance(
        dataclass_type, type
    ):
        raise TypeError(f"Expected a dataclass class, got {dataclass_type}")

    cached = _OVERLAY_TYPE_CACHE.get(dataclass_type)
    if cached is not None:
        return cached

    fields = [
        (
            field.name,
            _overlay_annotation(field.type),
            dataclasses.field(default=UNSET),
        )
        for field in dataclasses.fields(dataclass_type)
    ]
    overlay_cls = dataclasses.make_dataclass(
        f"{dataclass_type.__name__}Overlay",
        fields,
        frozen=True,
    )
    overlay_cls.__module__ = dataclass_type.__module__
    _OVERLAY_TYPE_CACHE[dataclass_type] = overlay_cls
    return overlay_cls


def overlay(dataclass_type: type[object], **values: object) -> DataclassInstance:
    """Build a dataclass overlay with unspecified fields set to ``UNSET``.

    The returned object is intended for ``apply_overrides``. It deliberately
    uses a generated overlay type so config classes with concrete defaults can
    be reused as sparse overlay shapes without changing those defaults.
    """
    overlay_cls = overlay_type(dataclass_type)
    field_names = {field.name for field in dataclasses.fields(overlay_cls)}
    unknown = set(values) - field_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise UntyroUnsupportedOverrideError(
            f"Overlay fields do not exist on {dataclass_type.__name__}: {names}"
        )

    return overlay_cls(**values)


def overrides(
    target: object,
    default: object = _MISSING,
    *,
    default_config: object = _MISSING,
    field_order: tuple[str, ...] | None = None,
) -> list[str]:
    """Return tyro argv tokens for CLI-representable differences.

    Unlike ``untyro(target)``, this function is intentionally strict. It emits
    only changed fields relative to ``default``/``default_config`` and raises
    ``UntyroUnsupportedOverrideError`` when a changed value cannot be expressed
    using tyro's current argv syntax.
    """
    if default is not _MISSING and default_config is not _MISSING:
        raise TypeError("Pass either default or default_config, not both")
    if default_config is not _MISSING:
        default = default_config
    if default is _MISSING:
        raise TypeError("A default config is required for override serialization")

    if dataclasses.is_dataclass(target) and not isinstance(target, type):
        return _dataclass_overrides(target, default, field_order=field_order)
    if isinstance(target, dict):
        return _dict_overrides(target, default)
    raise TypeError(f"Expected a dataclass instance or dict, got {type(target)}")


def _module_to_script_path(module: types.ModuleType) -> str:
    fpath = getattr(module, "__file__", None)
    if fpath is None:
        raise ValueError(f"Module {module.__name__} has no __file__ attribute")
    return os.path.relpath(fpath)


@overload
def untyro(args: object) -> list[str]: ...


@overload
def untyro(module: types.ModuleType, args: object) -> list[str]: ...


def untyro(module_or_args: object, args: object | None = None) -> list[str]:
    """Convert a dataclass instance or dict into tyro-compatible argv tokens.

    Single-argument dataclass form:
        ``untyro(Args(x=2))`` returns ``["--x", "2"]``.

    Single-argument dict form:
        ``untyro(dict(data=DataConfig(...)))`` returns prefixed flags such as
        ``["--data.field", "value"]``.

    Two-argument form:
        ``untyro(module, args)`` prepends the module file path relative to the
        current working directory.
    """
    if args is None:
        if isinstance(module_or_args, dict):
            return _dict_to_args(module_or_args)
        return _dataclass_to_args(module_or_args)

    if not isinstance(module_or_args, types.ModuleType):
        raise TypeError(f"First argument must be a module, got {type(module_or_args)}")
    path = _module_to_script_path(module_or_args)
    if isinstance(args, dict):
        return [path] + _dict_to_args(args)
    return [path] + _dataclass_to_args(args)


untyro.overrides = overrides  # type: ignore[attr-defined]
untyro.apply_overrides = apply_overrides  # type: ignore[attr-defined]
untyro.overlay = overlay  # type: ignore[attr-defined]
untyro.overlay_type = overlay_type  # type: ignore[attr-defined]
untyro.UNSET = UNSET  # type: ignore[attr-defined]
