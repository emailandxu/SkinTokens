from __future__ import annotations

from pathlib import Path


def load_rig_txt_payload(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    joints_by_name: dict[str, list[float]] = {}
    parents_by_child: dict[str, str] = {}
    root_name = None
    with resolved.open("r", encoding="utf-8") as rig_file:
        for line in rig_file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "joints":
                if len(parts) < 5:
                    raise ValueError(f"invalid joints line: {line.rstrip()}")
                x, y, z = (float(value) for value in parts[2:5])
                joints_by_name[parts[1]] = [x, -z, y]
            elif parts[0] == "root":
                if len(parts) < 2:
                    raise ValueError(f"invalid root line: {line.rstrip()}")
                root_name = parts[1]
            elif parts[0] == "hier":
                if len(parts) < 3:
                    raise ValueError(f"invalid hier line: {line.rstrip()}")
                parents_by_child[parts[2]] = parts[1]

    if not joints_by_name:
        raise ValueError(f"rig txt has no joints: {resolved}")
    if root_name is None:
        roots = set(joints_by_name) - set(parents_by_child)
        if len(roots) != 1:
            raise ValueError(f"rig txt must provide one root: {resolved}")
        root_name = next(iter(roots))
    if root_name not in joints_by_name:
        raise ValueError(f"root joint {root_name!r} is not declared in {resolved}")

    children_by_parent: dict[str, list[str]] = {
        name: [] for name in joints_by_name
    }
    for child, parent in parents_by_child.items():
        if child in joints_by_name and parent in joints_by_name:
            children_by_parent[parent].append(child)

    names: list[str] = []
    parents: list[int] = []

    def visit(name: str, parent: int) -> None:
        current = len(names)
        names.append(name)
        parents.append(parent)
        for child in children_by_parent.get(name, []):
            visit(child, current)

    visit(root_name, -1)
    if len(names) != len(joints_by_name):
        missing = sorted(set(joints_by_name) - set(names))
        raise ValueError(
            f"rig txt has joints disconnected from root {root_name!r}: {missing}"
        )
    return {
        "joints": [joints_by_name[name] for name in names],
        "parents": parents,
        "joint_names": names,
        "done": False,
    }
