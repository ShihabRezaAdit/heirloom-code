"""YAML config loading with `inherits:`, and the scientific config hash.

Rules (configs/README.md, decisions D-31 to D-34):
  * Only the scientific part of a config is hashed; engineering settings may
    change with hardware and are recorded in provenance instead.
  * The hash covers experiment_id, condition, seeds and the `scientific:` block.
  * `inherits:` paths are resolved relative to the file that names them, and
    child values override parent values key by key (deep merge).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .checksums import hash_obj

HASHED_TOP_LEVEL_KEYS = ("experiment_id", "condition", "seeds", "scientific")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_config(path: str | Path, _seen: tuple = ()) -> dict:
    """Load a YAML config, resolving `inherits:` recursively."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular inherits: {' -> '.join(map(str, _seen + (path,)))}")
    data = read_yaml(path)
    parent_ref = data.pop("inherits", None)
    if parent_ref:
        parent = load_config((path.parent / parent_ref).resolve(), _seen + (path,))
        parent.pop("_source", None)
        parent.pop("_chain", None)
        chain = parent.pop("_inherit_chain", [])
        data = deep_merge(parent, data)
        data["_inherit_chain"] = chain + [str(path)]
    else:
        data["_inherit_chain"] = [str(path)]
    data["_source"] = str(path)
    return data


def scientific_part(cfg: dict) -> dict:
    return {k: cfg.get(k) for k in HASHED_TOP_LEVEL_KEYS if k in cfg}


def config_hash(cfg: dict) -> str:
    """Full SHA-256 of the scientific part."""
    return hash_obj(scientific_part(cfg))


def config_hash8(cfg: dict) -> str:
    return config_hash(cfg)[:8]


def seed_list(cfg: dict) -> list[int]:
    """`seeds: 2` means [0, 1]; `seeds: [0, 1, 2]` is used as given."""
    seeds = cfg.get("seeds")
    if isinstance(seeds, int):
        return list(range(seeds))
    if isinstance(seeds, list) and all(isinstance(s, int) for s in seeds):
        return seeds
    raise ValueError(f"{cfg.get('_source')}: 'seeds' must be an int or a list of ints, got {seeds!r}")


def get(cfg: dict, dotted: str, default: Any = KeyError) -> Any:
    """cfg['a']['b'] via get(cfg, 'a.b'); raises KeyError with the full path if missing."""
    node: Any = cfg
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif default is KeyError:
            raise KeyError(f"{cfg.get('_source', '<config>')}: missing key '{dotted}'")
        else:
            return default
    return node
