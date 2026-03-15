from __future__ import annotations


def to_graph_node_name(identifier: str) -> str:
    return identifier.replace("::", "__")
