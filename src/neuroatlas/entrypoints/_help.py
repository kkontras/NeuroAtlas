"""Dynamic ``--help`` for the unified entrypoints.

``--help`` should answer "what can I actually pass?", not just "what flags
exist".  So the epilog is built from the live registry:

* bare ``--help`` lists the domains and dataset counts, every task and preset,
  and the model families;
* ``--dataset X --help`` additionally lists X's label modes, its default task,
  and **every key ``--set`` accepts for X, with its current default**.

That last list is the one that matters.  Dataset config is passed to the
datamodule as ``**kwargs``, so an unknown or misspelled key surfaces as
``TypeError: __init__() got an unexpected keyword argument`` from inside a
constructor — the same failure mode that makes the shipped quickstart config
fail on every cell.  Printing the accepted keys turns that into something you
can see before launching.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

_INDENT = "  "


def _wrap(items: Sequence[str], width: int = 76, indent: str = _INDENT) -> List[str]:
    if not items:
        return [f"{indent}(none)"]
    return textwrap.wrap(", ".join(items), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def peek_dataset(argv: Optional[Sequence[str]]) -> Optional[str]:
    """Find ``--dataset X`` in argv before argparse runs.

    argparse acts on ``--help`` the moment it sees it, so the epilog has to be
    built beforehand; this reads the flag directly.
    """
    if not argv:
        return None
    for i, token in enumerate(argv):
        if token == "--dataset" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--dataset="):
            return token.split("=", 1)[1]
    return None


def _datamodule_class(spec) -> Optional[type]:
    """Resolve the datamodule class behind a spec's factory.

    A dataset declared by its manifest names its class there
    (``spec.datamodule``, a dotted path). The other factories are ``def
    _create_x(**config): from .adapters.x import C; return C(**config)`` --
    the import is deliberately lazy, so the class is found by reading that
    import rather than by calling the factory.
    """
    import importlib

    dotted = ((spec.manifest or {}).get("spec") or {}).get("datamodule")
    if dotted:
        module_path, _, attr = str(dotted).rpartition(".")
        try:
            candidate = getattr(importlib.import_module(module_path), attr, None)
        except Exception:
            candidate = None
        if isinstance(candidate, type):
            return candidate
    try:
        source = textwrap.dedent(inspect.getsource(spec.datamodule_cls))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None

    module_name = getattr(spec.datamodule_cls, "__module__", None)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.names:
            continue
        target = node.names[0].name
        package = module_name.rsplit(".", 1)[0] if module_name else ""
        dotted = ("." * node.level) + (node.module or "")
        try:
            mod = importlib.import_module(dotted, package=package) if node.level \
                else importlib.import_module(node.module)
            candidate = getattr(mod, target, None)
        except Exception:
            continue
        if isinstance(candidate, type):
            return candidate
    return None


def dataset_set_keys(spec) -> Tuple[List[Tuple[str, Any]], Optional[str]]:
    """``--set`` keys for one dataset, as (key, default) pairs.

    Union of the manifest's ``runtime_defaults`` and the datamodule's
    ``__init__`` signature; the signature is authoritative because it is what
    raises on an unknown key.  Returns ``(pairs, note)``: *note* is set when
    the list is the dataset's defaults only (its class takes any key, or is
    not found), and says so in the header the help prints.
    """
    defaults: Dict[str, Any] = dict(spec.config_defaults)
    cls = _datamodule_class(spec)
    note = None
    if cls is None:
        note = "defaults"
        keys = sorted(defaults)
    else:
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            params = {}
        accepted = [name for name, p in params.items()
                    if name != "self" and p.kind not in
                    (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        if not accepted:
            note = "defaults"
            keys = sorted(defaults)
        else:
            keys = sorted(set(accepted) | set(defaults))
    pairs = []
    for key in keys:
        if key in defaults:
            pairs.append((key, defaults[key]))
        else:
            pairs.append((key, None))
    return pairs, note


def dataset_options(spec) -> Dict[str, List[str]]:
    """Allowed values per enumerable ``--set`` key, from the manifest.

    ``runtime_options`` records what the adapter/dataio actually validates, so
    ``--help`` can show the choices instead of only the current value.  Label
    modes fall back to ``labels.modes`` when no explicit options block exists.
    """
    manifest = spec.manifest or {}
    out: Dict[str, List[str]] = {}
    for key, entry in (manifest.get("runtime_options") or {}).items():
        values = (entry or {}).get("values")
        if values:
            out[key] = [str(v) for v in values]
    modes = list(((manifest.get("labels") or {}).get("modes") or {}))
    if modes and "label_mode" not in out:
        out["label_mode"] = modes

    # Never advertise a key the datamodule does not take.  Several datasets
    # declare label modes that their loader selects internally rather than via
    # a ``label_mode`` argument (shhs, sleep_edf, parkinson), and TUAB's
    # adapter exposes ``montage_filter`` rather than ``montage`` — offering
    # those would send users straight into the ``unexpected keyword argument``
    # failure this help exists to prevent.
    accepted = {key for key, _default in dataset_set_keys(spec)[0]}
    return {key: values for key, values in out.items() if key in accepted}


#: Keys the help does not list: settings with one value the user never
#: chooses (the PhysioEx readers' preprocessing pipeline name).
_HIDDEN_KEYS = frozenset({"pipeline_name"})

def _format_default(value: Any) -> str:
    if value is None:
        return ""
    text = repr(value)
    return text if len(text) <= 46 else text[:43] + "..."


def dataset_section(slug: str) -> List[str]:
    """The ``--dataset X --help`` block."""
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    try:
        spec = load_dataset_spec(slug)
    except KeyError:
        from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

        near = [s.slug for s in dataset_specs() if slug.lower() in s.slug.lower()][:8]
        lines = [f"no dataset named {slug!r}."]
        if near:
            lines += ["", "did you mean:"] + _wrap(near)
        lines += ["", "`neuroatlas list datasets --all` names them all."]
        return lines

    manifest = spec.manifest or {}
    lines = [f"dataset: {slug}"]
    if manifest.get("name"):
        tags = [t for t in (manifest.get("domain"),
                            "in the paper" if manifest.get("paper_dataset") else None) if t]
        lines.append(f"{_INDENT}{manifest['name']}" + (f"  [{', '.join(tags)}]" if tags else ""))
    acq = manifest.get("acquisition") or {}
    if acq.get("kind"):
        lines.append(f"{_INDENT}source: {acq['kind']}"
                     f"{' ' + str(acq.get('ref')) if acq.get('ref') else ''}")
    lines.append(f"{_INDENT}default task: {spec.default_task}")

    labels = manifest.get("labels") or {}
    modes = labels.get("modes") or {}
    default_gran = labels.get("granularity")
    if modes:
        lines += ["", "  label modes (--set label_mode=...):"]
        # "binary" alone does not say binary of what. Printing the granularity
        # is the difference between seizure-vs-background per 10 s window and
        # abnormal-vs-normal per recording.
        width = max(len(m) for m in modes)
        for mode, block in modes.items():
            gran = (block or {}).get("granularity") or default_gran
            marker = " (default)" if mode == labels.get("default") else ""
            lines.append(f"{_INDENT * 2}{mode:<{width}}  one label per "
                         f"{gran}{marker}")

    also = list((labels.get("also_implemented") or []))
    if also:
        lines += [f"{_INDENT * 2}also in the reader, not used by any benchmark:"] \
            + _wrap(also, indent=_INDENT * 3)

    splits = manifest.get("splits") or {}
    if splits.get("n_folds"):
        by = f" by {splits['grouping']}" if splits.get("grouping") else ""
        # (how the folds were made is in the dataset's manifest, for whoever
        # maintains them; the help says only what a user relies on)
        source = ("fixed by a fold file that ships with neuroatlas"
                  if splits.get("manifest") else "drawn from the data when it is read")
        lines += ["", f"  splits: {splits['n_folds']} folds{by}, {source}"]

    backends = manifest.get("backends") or {}
    if backends:
        lines += ["", "  backends (--set backend=...):"]
        for name, block in backends.items():
            block = block or {}
            reads = f" reads {block['reads']}" if block.get("reads") else ""
            lines.append(f"{_INDENT * 2}{name:<6}{reads}".rstrip())
            # The --set table below is the DEFAULT backend's. Say which keys
            # each other backend adds and which it refuses, or a user reads the
            # table as if it applied to all of them.
            base = dict(spec.config_defaults)
            defaults = block.get("defaults") or {}
            extra = sorted(k for k in defaults if k not in base)
            changed = sorted(f"{k}={defaults[k]!r}" for k in defaults if k in base)
            gone = sorted(block.get("unsupported") or {})
            if extra:
                lines.append(f"{_INDENT * 3}adds:    {', '.join(extra)}")
            if changed:
                lines.append(f"{_INDENT * 3}changes: {', '.join(changed)}")
            if gone:
                lines.append(f"{_INDENT * 3}ignores: {', '.join(gone)}")

    pairs, note = dataset_set_keys(spec)
    # settings the user does not choose between (an internal pipeline name)
    pairs = [(key, default) for key, default in pairs if key not in _HIDDEN_KEYS]
    options = dataset_options(spec)
    if note:
        lines += ["", f"  --set keys accepted by {slug} (the dataset's defaults):"]
    else:
        lines += ["", f"  --set keys accepted by {slug} (default value; the allowed values "
                      "where there is a fixed set):"]
    width = max((len(k) for k, _ in pairs), default=10)
    pad = _INDENT * 2
    for key, default in pairs:
        shown = _format_default(default)
        line = f"{pad}{key:<{width}}  {shown}"
        choices = options.get(key)
        if choices and len(choices) < 2:
            choices = None          # "one of: x" for a single value is noise
        if not choices:
            lines.append(line.rstrip())
            continue
        rendered = "one of: " + ", ".join(choices)
        if len(line) + len(rendered) + 2 <= 96:
            lines.append(f"{line}  {rendered}")
        else:
            lines.append(line.rstrip())
            lines += textwrap.wrap(rendered, width=92,
                                   initial_indent=pad + " " * (width + 2),
                                   subsequent_indent=pad + " " * (width + 2))
    return lines


def overview_section(*, show_models: bool = True) -> List[str]:
    """The bare ``--help`` block: the paper's datasets by domain, the others,
    and -- for the commands that take them (``show_models``: embed and
    probe) -- the tasks, presets and models."""
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    specs = list(dataset_specs())
    by_domain: Dict[str, List[str]] = {}
    others: List[str] = []
    generated = 0
    for spec in specs:
        manifest = spec.manifest or {}
        domain = manifest.get("domain")
        if not domain:
            generated += 1
        elif manifest.get("paper_dataset"):
            by_domain.setdefault(domain, []).append(spec.slug)
        else:
            others.append(spec.slug)

    n_paper = sum(len(v) for v in by_domain.values())
    lines = [f"the paper's datasets (--dataset), {n_paper}:"]
    for domain in sorted(by_domain):
        slugs = sorted(by_domain[domain])
        lines.append(f"{_INDENT}{domain} ({len(slugs)}):")
        lines += _wrap(slugs, indent=_INDENT * 2)
    if others or generated:
        also = sorted(others) + ([f"{generated} MOABB datasets"] if generated else [])
        text = ", ".join(also[:-1]) + (" and " if len(also) > 1 else "") + also[-1]
        lines += textwrap.wrap(f"also readable, not in the paper: {text}", width=78,
                               initial_indent=_INDENT, subsequent_indent=_INDENT * 2)
    lines.append(f"{_INDENT}`neuroatlas list datasets --all` lists all {len(specs)}")
    if not show_models:
        return lines

    try:
        from neuroatlas.entrypoints.probe import available_tasks
        from neuroatlas.benchmarking_helpers.registry.discovery import task_specs

        lines += ["", "tasks (--task):"]
        lines += _wrap(sorted(s.slug for s in task_specs()))
        presets = available_tasks()
        if presets:
            lines += [f"{_INDENT}task presets:"]
            lines += _wrap(presets, indent=_INDENT * 2)
    except Exception:
        pass

    if show_models:
        try:
            from neuroatlas.benchmarking_helpers.registry.discovery import model_specs

            lines += ["", "models (--models, comma-separated families or checkpoint ids):"]
            lines += _wrap(sorted(s.slug for s in model_specs()))
        except Exception:
            pass

    lines += ["", "for one dataset's --set keys and label modes, add --dataset DATASET --help:",
              f"{_INDENT}neuroatlas embed --dataset hmc --help"]
    return lines


def build_epilog(argv: Optional[Sequence[str]], *, show_models: bool = True) -> str:
    """Epilog for a parser, tailored to ``--dataset`` when argv names one."""
    slug = peek_dataset(argv)
    try:
        lines = dataset_section(slug) if slug else overview_section(show_models=show_models)
    except Exception as exc:                      # help must never crash the CLI
        lines = [f"(could not build the dynamic help: {exc})"]
    return "\n".join(lines)
