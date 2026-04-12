import re

from huggingface_hub import HfApi


def _natural_sort_key(name: str) -> tuple:
    """Sort key that handles branch names like 'stage2-step-3480000' correctly.

    Splits the name into (prefix, number) pairs so that:
    - 'stage1-step-3440000' < 'stage2-step-3480000' (prefix order)
    - 'stage1-step-80000' < 'stage1-step-120000'   (numeric order within prefix)
    - 'longctx-step125' < 'longctx-step1000'        (numeric order)
    - 'step100000-tokens420B' < 'step950000-tokens3990B'
    """
    parts = re.split(r"(\d+)", name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def list_branches(model_id: str) -> list[str]:
    """List all branch names for a HuggingFace model repo, sorted alphabetically."""
    api = HfApi()
    refs = api.list_repo_refs(model_id)
    branches = sorted(b.name for b in refs.branches)
    return branches


def sort_checkpoints(names: list[str]) -> list[str]:
    """Sort checkpoint branch names in natural training order."""
    return sorted(names, key=_natural_sort_key)


def resolve_checkpoints(
    model_id: str,
    *,
    last: int | None = None,
    total: int | None = None,
    names: list[str] | None = None,
) -> list[str]:
    """Resolve which checkpoints (branches) to evaluate.

    Exactly one of last, total, or names must be provided.
    - last N: take the last N branches (naturally sorted by training order)
    - total T: take T evenly spaced branches from the naturally sorted list
    - names: use these exact branch names (in the given order)
    """
    if sum(x is not None for x in (last, total, names)) != 1:
        raise ValueError("Exactly one of --last, --total, or --names must be specified")

    if names is not None:
        return names

    branches = list_branches(model_id)
    # Exclude 'main' from checkpoint branches for --last/--total
    checkpoint_branches = [b for b in branches if b != "main"]

    if not checkpoint_branches:
        print(f"No checkpoint branches found for {model_id}, falling back to 'main'")
        return ["main"]

    checkpoint_branches = sort_checkpoints(checkpoint_branches)

    if last is not None:
        return checkpoint_branches[-last:]

    # total: evenly spaced
    if total >= len(checkpoint_branches):
        return checkpoint_branches
    step = (len(checkpoint_branches) - 1) / (total - 1)
    indices = [round(i * step) for i in range(total)]
    return [checkpoint_branches[i] for i in indices]
