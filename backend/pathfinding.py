"""
pathfinding.py – Dijkstra na grafu budovy.
Graf je načítán z data/building_graph.json.
"""

import json
import heapq
from pathlib import Path
from typing import Optional

GRAPH_PATH = Path(__file__).parent / "data" / "building_graph.json"


def load_graph(path: Path = GRAPH_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_adjacency(graph: dict) -> dict[str, list[tuple[str, float]]]:
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in graph["nodes"]}
    for edge in graph["edges"]:
        a, b, w = edge["from"], edge["to"], edge["weight"]
        adj[a].append((b, w))
        adj[b].append((a, w))  # neorientovaný graf
    return adj


def dijkstra(adj: dict, start: str) -> tuple[dict[str, float], dict[str, Optional[str]]]:
    dist: dict[str, float] = {n: float("inf") for n in adj}
    prev: dict[str, Optional[str]] = {n: None for n in adj}
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def reconstruct_path(prev: dict, end: str) -> list[str]:
    path = []
    cur: Optional[str] = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def find_nearest_exit(start_node: str) -> dict:
    """
    Vrátí nejkratší cestu k nejbližšímu exitu z daného uzlu.
    Returns:
        {
            "start": str,
            "exit": str,
            "distance": float,
            "path": list[str],
            "path_labels": list[str],
            "instructions": list[str]
        }
    """
    graph = load_graph()
    adj = build_adjacency(graph)
    nodes = graph["nodes"]

    if start_node not in adj:
        raise ValueError(f"Uzel '{start_node}' neexistuje v grafu budovy.")

    dist, prev = dijkstra(adj, start_node)

    # Najdi nejbližší exit
    exits = [n for n, meta in nodes.items() if meta.get("type") == "exit"]
    if not exits:
        raise RuntimeError("Graf neobsahuje žádný exit.")

    nearest = min(exits, key=lambda e: dist[e])
    if dist[nearest] == float("inf"):
        raise RuntimeError(f"Exit '{nearest}' není dosažitelný z '{start_node}'.")

    path = reconstruct_path(prev, nearest)
    labels = [nodes[n]["label"] for n in path]
    instructions = _generate_instructions(path, nodes)

    return {
        "start": start_node,
        "exit": nearest,
        "distance": dist[nearest],
        "path": path,
        "path_labels": labels,
        "instructions": instructions,
    }


def _generate_instructions(path: list[str], nodes: dict) -> list[str]:
    """Lidsky čitelné pokyny krok po kroku."""
    if not path:
        return []
    instructions = [f"Nacházíte se v: {nodes[path[0]]['label']}"]
    for i in range(1, len(path)):
        node = nodes[path[i]]
        t = node.get("type", "room")
        label = node["label"]
        if t == "exit":
            instructions.append(f"✅ Opusťte budovu přes: {label}")
        elif t == "stair":
            instructions.append(f"🔼 Přejděte na schodiště: {label}")
        else:
            instructions.append(f"➡️  Pokračujte do: {label}")
    return instructions


if __name__ == "__main__":
    result = find_nearest_exit("ucebna_201")
    for step in result["instructions"]:
        print(step)
    print(f"\nCelková vzdálenost: {result['distance']}")
