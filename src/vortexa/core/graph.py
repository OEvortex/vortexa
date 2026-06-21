"""Repository Knowledge Graph — in-memory graph with LMDB persistence.

Stores relationships between files, classes, functions, and symbols.
No database server required — uses adjacency lists in LMDB following
the same persistence pattern as VectorStore.

V2 additions (inspired by JARVIS kg/serve.py):
- BFS/DFS traversal with depth control and hub-thresholding
- Query-aware node scoring (IDF-weighted term matching)
- God-node (most connected) analysis
- Structural-relation filtering for noise suppression
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import lmdb

from vortexa.core.types import (
    EdgeType,
    GraphEdge,
    GraphEdgeInfo,
    GraphNode,
    GraphNodeInfo,
    GraphPath,
    GraphTraversalMode,
    SymbolKind,
)

logger = logging.getLogger(__name__)

_LMDB_MAP_SIZE = 64 * 1024 * 1024  # 64 MB — graph is metadata, not vectors

# Search/tokenization helpers used by query-aware scoring. Mirrors JARVIS's
# _search_tokens from kg/serve.py so that query-aware expansion feels
# consistent across both codebases.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how if in is it "
    "not of on or the to was what when where which who why with".split()
)


class KnowledgeGraph:
    """Repository-scale knowledge graph.

    Nodes represent files, classes, functions, methods, and symbols.
    Edges represent relationships: imports, calls, extends, defines, tests, etc.

    Design:
    - In-memory adjacency lists for O(1) traversal during search
    - LMDB persistence for loading/saving between sessions
    - No external dependencies beyond lmdb
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._out_edges: dict[str, list[GraphEdge]] = defaultdict(list)
        self._in_edges: dict[str, list[GraphEdge]] = defaultdict(list)

        # Derived indexes for fast lookup
        self._file_to_nodes: dict[str, list[str]] = defaultdict(list)  # file_path -> node_ids
        self._kind_index: dict[SymbolKind, list[str]] = defaultdict(list)  # kind -> node_ids
        self._name_index: dict[str, list[str]] = defaultdict(list)  # label -> node_ids

    # ── Node operations ──────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        kind: SymbolKind,
        label: str,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GraphNode:
        """Add a node to the graph. Replaces existing node with same id."""
        node = GraphNode(
            id=node_id,
            kind=kind,
            label=label,
            file_path=file_path,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        self._kind_index[kind].append(node_id)
        self._name_index[label].append(node_id)
        if file_path:
            self._file_to_nodes[file_path].append(node_id)
        return node

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return
        # Remove from indexes
        if node.label in self._name_index:
            try:
                self._name_index[node.label].remove(node_id)
            except ValueError:
                pass
        if node.kind in self._kind_index:
            try:
                self._kind_index[node.kind].remove(node_id)
            except ValueError:
                pass
        if node.file_path and node.file_path in self._file_to_nodes:
            try:
                self._file_to_nodes[node.file_path].remove(node_id)
            except ValueError:
                pass
        # Remove edges
        self._out_edges.pop(node_id, None)
        # Remove incoming edges pointing to this node
        for source_id, edges in list(self._in_edges.items()):
            self._in_edges[source_id] = [e for e in edges if e.target != node_id]
        # Remove from in_edges of others
        for source_id in list(self._out_edges.keys()):
            self._out_edges[source_id] = [e for e in self._out_edges[source_id] if e.target != node_id]
        self._in_edges.pop(node_id, None)

    # ── Edge operations ──────────────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> GraphEdge | None:
        """Add a directed edge. Returns None if either node doesn't exist."""
        if source not in self._nodes or target not in self._nodes:
            return None
        edge = GraphEdge(
            source=source,
            target=target,
            type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self._out_edges[source].append(edge)
        self._in_edges[target].append(edge)
        return edge

    def add_undirected_edge(
        self,
        a: str,
        b: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an undirected edge (two directed edges)."""
        self.add_edge(a, b, edge_type, weight, metadata)
        self.add_edge(b, a, edge_type, weight, metadata)

    # ── Query operations ─────────────────────────────────────────────────

    def neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        direction: str = "out",
    ) -> list[GraphEdge]:
        """Get neighboring edges of a node, optionally filtered by edge type.

        :param node_id: The node to query.
        :param edge_type: Optional filter by edge type.
        :param direction: 'out' (outgoing), 'in' (incoming), or 'both'.
        """
        result: list[GraphEdge] = []
        if direction in ("out", "both"):
            for edge in self._out_edges.get(node_id, []):
                if edge_type is None or edge.type == edge_type:
                    result.append(edge)
        if direction in ("in", "both"):
            for edge in self._in_edges.get(node_id, []):
                if edge_type is None or edge.type == edge_type:
                    result.append(edge)
        return result

    def expand(
        self,
        node_ids: list[str],
        depth: int = 1,
        edge_types: list[EdgeType] | None = None,
    ) -> list[GraphEdge]:
        """Expand outward from a set of nodes up to `depth` levels.

        Returns all unique edges discovered during expansion.
        Edge deduplication uses (source, target, type) as the key.
        """
        discovered: set[str] = set(node_ids)
        frontier: list[str] = list(node_ids)
        all_edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for _ in range(depth):
            next_frontier: list[str] = []
            for nid in frontier:
                for edge in self._out_edges.get(nid, []):
                    if edge_types and edge.type not in edge_types:
                        continue
                    key = (edge.source, edge.target, edge.type.value)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        all_edges.append(edge)
                    if edge.target not in discovered:
                        discovered.add(edge.target)
                        next_frontier.append(edge.target)
                for edge in self._in_edges.get(nid, []):
                    if edge_types and edge.type not in edge_types:
                        continue
                    key = (edge.source, edge.target, edge.type.value)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        all_edges.append(edge)
                    if edge.source not in discovered:
                        discovered.add(edge.source)
                        next_frontier.append(edge.source)
            frontier = next_frontier

        return all_edges

    def shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> list[GraphEdge] | None:
        """BFS shortest path between two nodes."""
        if source not in self._nodes or target not in self._nodes:
            return None

        visited: set[str] = {source}
        queue: deque[tuple[str, list[GraphEdge]]] = deque([(source, [])])

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for edge in self._out_edges.get(current, []):
                if edge.target == target:
                    return path + [edge]
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge]))
        return None

    # ── Index lookups ────────────────────────────────────────────────────

    def find_nodes_by_name(self, name: str) -> list[GraphNode]:
        """Find nodes by label (symbol name, file name, etc.)."""
        return [self._nodes[nid] for nid in self._name_index.get(name, [])]

    def find_nodes_by_kind(self, kind: SymbolKind) -> list[GraphNode]:
        """Find all nodes of a given kind."""
        return [self._nodes[nid] for nid in self._kind_index.get(kind, [])]

    def find_nodes_in_file(self, file_path: str) -> list[GraphNode]:
        """Find all nodes defined in a file."""
        return [self._nodes[nid] for nid in self._file_to_nodes.get(file_path, [])]

    def find_file_node(self, file_path: str) -> GraphNode | None:
        """Find the file node for a given file path."""
        for nid in self._kind_index.get(SymbolKind.FILE, []):
            node = self._nodes[nid]
            if node.label == file_path:
                return node
        return None

    # ── V2: Query-aware scoring & traversal ─────────────────────────────────

    def _node_label_lower(self, node_id: str) -> str:
        """Return lowercased label for a node (safe default)."""
        node = self._nodes.get(node_id)
        return (node.label if node else node_id).lower()

    def _node_file_path_lower(self, node_id: str) -> str:
        """Return lowercased source_file for a node."""
        node = self._nodes.get(node_id)
        return (node.file_path or "").lower() if node else ""

    def _node_kind_str(self, node_id: str) -> str:
        """Return string kind for a node (defaults to 'unknown')."""
        node = self._nodes.get(node_id)
        return node.kind.value if node else "unknown"

    def _query_tokens(self, query: str) -> list[str]:
        """Tokenize a query into lowercased word tokens, dropping stopwords."""
        if not query:
            return []
        return [
            tok for tok in _TOKEN_RE.findall(query.lower())
            if tok and tok not in _STOPWORDS
        ]

    def _node_idf(self, terms: list[str]) -> dict[str, float]:
        """Compute IDF weights for query terms across all node labels.

        Common terms like 'error' that match hundreds of nodes get low
        weights; rare identifiers like 'FooBarService' get high weights.
        """
        if not terms or not self._nodes:
            return {t: 1.0 for t in terms}
        n = max(len(self._nodes), 1)
        idf: dict[str, float] = {}
        for t in set(terms):
            df = sum(
                1 for nid in self._nodes
                if t in self._node_label_lower(nid) or t in self._node_file_path_lower(nid)
            )
            idf[t] = math.log(1.0 + n / (1.0 + df))
        return idf

    # Match-tier bonuses (mirrors JARVIS's _EXACT_MATCH_BONUS etc.).
    _EXACT_BONUS = 1000.0
    _PREFIX_BONUS = 100.0
    _SUBSTRING_BONUS = 1.0
    _SOURCE_BONUS = 0.5

    def score_nodes_against_query(self, query: str) -> list[tuple[float, str]]:
        """Score every node against the query, ranked by relevance.

        Three-tier precedence: exact > prefix > substring. Per-term IDF
        weights the scores so common terms don't drown out rare identifiers.

        Returns a list of (score, node_id) tuples, sorted descending.
        """
        terms = self._query_tokens(query)
        if not terms:
            return []
        idf = self._node_idf(terms)
        scored: list[tuple[float, str]] = []
        for nid in self._nodes:
            label = self._node_label_lower(nid)
            source = self._node_file_path_lower(nid)
            score = 0.0
            for t in terms:
                w = idf.get(t, 1.0)
                if t == label:
                    score += self._EXACT_BONUS * w
                elif label.startswith(t):
                    score += self._PREFIX_BONUS * w
                elif t in label:
                    score += self._SUBSTRING_BONUS * w
                if t in source:
                    score += self._SOURCE_BONUS * w
            if score > 0:
                # Light penalty for non-code nodes (rationale/document/concept).
                # Code is the default graph_query target; doc/rationale are
                # breadcrumbs that should never outrank real symbols.
                kind = self._node_kind_str(nid)
                if kind == "code":
                    pass
                elif kind == "rationale":
                    score *= 0.1
                elif kind == "document":
                    score *= 0.05
                else:
                    score *= 0.5
                scored.append((score, nid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def pick_seeds(
        self,
        scored: list[tuple[float, str]],
        max_k: int = 3,
        gap_ratio: float = 0.2,
    ) -> list[str]:
        """Pick top BFS/DFS seed nodes, stopping when score drops too far below the top.

        Prevents high-frequency noise terms from stealing seed slots from a
        dominant identifier match. When FooBarService scores 1000 and
        noise terms score 1.0, only FooBarService is seeded.
        """
        if not scored:
            return []
        top_score = scored[0][0]
        seeds: list[str] = []
        for score, nid in scored[:max_k]:
            if seeds and score < top_score * gap_ratio:
                break
            seeds.append(nid)
        return seeds

    def _hub_threshold(self) -> int:
        """Compute the hub threshold for traversal (p99 degree, floored at 50).

        Hubs above this threshold are not expanded through as transit during
        BFS/DFS — they're treated as visited, so the traversal explores the
        query-relevant neighbourhood instead of fanning out through Path/str/
        bool and other infrastructure noise that floods real codebases.
        """
        if not self._nodes:
            return 50
        degrees = sorted(len(self._out_edges.get(nid, [])) + len(self._in_edges.get(nid, []))
                         for nid in self._nodes)
        p99_idx = int(len(degrees) * 0.99)
        return max(50, degrees[p99_idx])

    def bfs_traverse(
        self,
        start_nodes: list[str],
        depth: int = 3,
        relation_filter: set[str] | None = None,
    ) -> tuple[set[str], list[tuple[str, str, str]]]:
        """Breadth-first traversal from seed nodes up to `depth` hops.

        Returns (visited_node_ids, edges_seen).
        Each edge is (source_label, relation, target_label).

        High-degree hubs are not expanded as transit (only the seeds
        themselves are explored) — keeps the traversal focused on the
        query-relevant neighbourhood.

        :param relation_filter: if set, only edges whose EdgeType.value is in
            the filter set are traversed. None = traverse all relations.
        """
        if not start_nodes:
            return set(), []
        hub_threshold = self._hub_threshold()
        seed_set = set(start_nodes)
        visited: set[str] = set(start_nodes)
        frontier = list(start_nodes)
        edges: list[tuple[str, str, str]] = []
        for _ in range(max(0, depth)):
            next_frontier: list[str] = []
            for nid in frontier:
                # Don't expand through hubs except for seeds themselves
                if nid not in seed_set and self.degree_of(nid) >= hub_threshold:
                    continue
                for edge in self._out_edges.get(nid, []):
                    if relation_filter and edge.type.value not in relation_filter:
                        continue
                    if edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.append(edge.target)
                        edges.append((
                            self._node_label_lower(nid),
                            edge.type.value,
                            self._node_label_lower(edge.target),
                        ))
            frontier = next_frontier
            if not frontier:
                break
        return visited, edges

    def dfs_traverse(
        self,
        start_nodes: list[str],
        depth: int = 3,
        relation_filter: set[str] | None = None,
    ) -> tuple[set[str], list[tuple[str, str, str]]]:
        """Depth-first traversal from seed nodes up to `depth` hops.

        Returns (visited_node_ids, edges_seen).
        Each edge is (source_label, relation, target_label).
        """
        if not start_nodes:
            return set(), []
        hub_threshold = self._hub_threshold()
        seed_set = set(start_nodes)
        visited: set[str] = set()
        edges: list[tuple[str, str, str]] = []
        stack: list[tuple[str, int]] = [(n, 0) for n in reversed(start_nodes)]
        while stack:
            node, d = stack.pop()
            if node in visited or d > depth:
                continue
            visited.add(node)
            if node not in seed_set and self.degree_of(node) >= hub_threshold:
                continue
            for edge in self._out_edges.get(node, []):
                if relation_filter and edge.type.value not in relation_filter:
                    continue
                if edge.target not in visited:
                    stack.append((edge.target, d + 1))
                    edges.append((
                        self._node_label_lower(node),
                        edge.type.value,
                        self._node_label_lower(edge.target),
                    ))
        return visited, edges

    def degree_of(self, node_id: str) -> int:
        """Total degree (in + out) for a single node."""
        return len(self._out_edges.get(node_id, [])) + len(self._in_edges.get(node_id, []))

    def god_nodes(self, top_n: int = 10) -> list[GraphNodeInfo]:
        """Return the top-N most connected real entities (most-connected nodes).

        File-level hub nodes are excluded — they accumulate import/contains
        edges mechanically and don't represent meaningful abstractions.
        """
        if not self._nodes:
            return []
        degrees = sorted(
            ((self.degree_of(nid), nid) for nid in self._nodes),
            key=lambda x: x[0],
            reverse=True,
        )
        result: list[GraphNodeInfo] = []
        for _deg, nid in degrees:
            node = self._nodes[nid]
            if self._is_file_hub_node(node):
                continue
            result.append(GraphNodeInfo(
                id=nid,
                label=node.label,
                kind=node.kind.value,
                file_path=node.file_path,
                degree=self.degree_of(nid),
                community=None,
            ))
            if len(result) >= top_n:
                break
        return result

    def _is_file_hub_node(self, node: GraphNode) -> bool:
        """File-level hubs and method stubs aren't real abstractions."""
        if node.kind == SymbolKind.FILE:
            return True
        if not node.label:
            return False
        if node.file_path:
            if node.label == Path(node.file_path).name:
                return True
        if node.label.startswith(".") and node.label.endswith("()"):
            return True
        return False

    def find_node(self, label: str) -> list[str]:
        """Find node IDs whose label or ID matches the search term.

        Three-tier precedence: exact match, then prefix match, then
        substring match. Code nodes surface before document/rationale nodes
        so a search for 'MCP' returns the real class before prose matches.
        """
        if not label:
            return []
        term = " ".join(self._query_tokens(label)).lower()
        if not term:
            return []

        def _type_priority(node: GraphNode) -> int:
            return {"code": 0, "rationale": 1, "document": 2}.get(node.kind.value, 3)

        def _rank(nid: str) -> tuple[int, int, int]:
            node = self._nodes[nid]
            return (_type_priority(node), -self.degree_of(nid), len(node.label))

        exact: list[str] = []
        prefix: list[str] = []
        substring: list[str] = []
        for nid in self._nodes:
            node = self._nodes[nid]
            norm_label = node.label.lower().rstrip("()")
            nid_lower = nid.lower()
            if term == norm_label or term == nid_lower:
                exact.append(nid)
            elif norm_label.startswith(term) or nid_lower.startswith(term):
                prefix.append(nid)
            elif term in norm_label:
                substring.append(nid)
        exact.sort(key=_rank)
        prefix.sort(key=_rank)
        substring.sort(key=_rank)
        return exact + prefix + substring

    def get_node_info(self, label: str) -> GraphNodeInfo | None:
        """Get details about a single node by label or ID."""
        matches = self.find_node(label)
        if not matches:
            return None
        nid = matches[0]
        node = self._nodes[nid]
        return GraphNodeInfo(
            id=nid,
            label=node.label,
            kind=node.kind.value,
            file_path=node.file_path,
            degree=self.degree_of(nid),
            community=None,
        )

    def get_neighbors(self, label: str) -> list[GraphEdgeInfo]:
        """Return incoming and outgoing edges for a node, matched by label.

        Each edge includes the relation type and direction. Empty list if the
        node isn't found.
        """
        matches = self.find_node(label)
        if not matches:
            return []
        nid = matches[0]
        edges: list[GraphEdgeInfo] = []
        for edge in self._out_edges.get(nid, []):
            edges.append(GraphEdgeInfo(
                source=nid,
                target=edge.target,
                relation=edge.type.value,
                direction="out",
            ))
        for edge in self._in_edges.get(nid, []):
            edges.append(GraphEdgeInfo(
                source=edge.source,
                target=nid,
                relation=edge.type.value,
                direction="in",
            ))
        return edges

    def shortest_path_between(
        self,
        source_label: str,
        target_label: str,
        max_hops: int = 8,
    ) -> GraphPath | None:
        """Find the shortest path between two nodes (matched by label).

        BFS over the undirected view of the graph. Returns None if no path
        exists, or if either endpoint can't be resolved.

        :param max_hops: Refuse to return paths longer than this. Defaults to
            8 to keep agent context compact — longer paths are usually noise.
        """
        sources = self.find_node(source_label)
        targets = self.find_node(target_label)
        if not sources or not targets:
            return None
        src, tgt = sources[0], targets[0]
        if src == tgt:
            return GraphPath(source=src, target=tgt, hops=0, segments=())

        visited: set[str] = {src}
        # Each queue entry: (current_node, path_segments). segments are
        # (source_label, relation, target_label) so the caller can render
        # the path without re-querying the graph.
        queue: deque[tuple[str, tuple[tuple[str, str, str], ...]]] = deque(
            [(src, ())]
        )
        while queue:
            current, segs = queue.popleft()
            if len(segs) >= max_hops:
                continue
            for edge in self._out_edges.get(current, []):
                next_segs = segs + ((
                    self._node_label_lower(current),
                    edge.type.value,
                    self._node_label_lower(edge.target),
                ),)
                if edge.target == tgt:
                    return GraphPath(
                        source=src,
                        target=tgt,
                        hops=len(next_segs),
                        segments=next_segs,
                    )
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, next_segs))
            # Also traverse incoming edges (treat graph as undirected)
            for edge in self._in_edges.get(current, []):
                if edge.source == tgt:
                    next_segs = segs + ((
                        self._node_label_lower(current),
                        edge.type.value,
                        self._node_label_lower(edge.source),
                    ),)
                    return GraphPath(
                        source=src,
                        target=tgt,
                        hops=len(next_segs),
                        segments=next_segs,
                    )
                if edge.source not in visited:
                    visited.add(edge.source)
                    queue.append((edge.source, segs + ((
                        self._node_label_lower(current),
                        edge.type.value,
                        self._node_label_lower(edge.source),
                    ),)))
        return None

    def query_graph(
        self,
        question: str,
        mode: GraphTraversalMode = GraphTraversalMode.BFS,
        depth: int = 3,
        relation_filter: list[str] | None = None,
        token_budget: int = 2000,
    ) -> str:
        """Query the knowledge graph with BFS or DFS and render as text.

        Auto-picks seed nodes via `score_nodes_against_query` +
        `pick_seeds`, then traverses with the requested mode up to `depth`
        hops. Output is rendered as compact text capped at ~token_budget
        characters (chars / 3 ≈ tokens).

        :param relation_filter: Edge-relation whitelist. None = use the
            default filter (suppresses type-annotation noise). Pass an
            explicit list (including empty) to override.
        """
        effective_filter = (
            set(relation_filter) if relation_filter is not None
            else None  # no filter applied at traversal level — let edges flow
        )
        terms = self._query_tokens(question)
        scored = self.score_nodes_against_query(question)
        seeds = self.pick_seeds(scored)
        if not seeds:
            return "No matching nodes found."
        if mode == GraphTraversalMode.DFS:
            visited, edges = self.dfs_traverse(seeds, depth=depth, relation_filter=effective_filter)
        else:
            visited, edges = self.bfs_traverse(seeds, depth=depth, relation_filter=effective_filter)

        lines: list[str] = []
        seed_labels = [self._node_label_lower(n) for n in seeds]
        lines.append(
            f"Traversal: {mode.value.upper()} depth={depth} | "
            f"Start: {seed_labels} | "
            f"{len(visited)} nodes, {len(edges)} edges"
        )
        lines.append("")

        # Render seeds first (always visible), then neighbours by degree
        seen_in_output: set[str] = set()
        for nid in seeds:
            lines.append(self._format_node_line(nid))
            seen_in_output.add(nid)
        other_nodes = sorted(
            visited - seen_in_output,
            key=lambda n: self.degree_of(n),
            reverse=True,
        )
        for nid in other_nodes:
            lines.append(self._format_node_line(nid))

        if edges:
            lines.append("")
            for src_label, relation, tgt_label in edges[:max(1, token_budget // 4)]:
                lines.append(
                    f"  {src_label} --[{relation}]--> {tgt_label}"
                )

        output = "\n".join(lines)
        char_budget = token_budget * 3
        if len(output) > char_budget:
            cut_at = output[:char_budget].rfind("\n")
            cut_at = cut_at if cut_at > 0 else char_budget
            output = output[:cut_at] + "\n... (truncated by token budget)"
        return output

    def _format_node_line(self, node_id: str) -> str:
        """Render one node as a single descriptive line."""
        node = self._nodes[node_id]
        return (
            f"NODE {node.label} "
            f"[kind={node.kind.value} src={node.file_path or ''}]"
        )

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        total = sum(len(edges) for edges in self._out_edges.values())
        return total

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "by_kind": {k.value: len(v) for k, v in self._kind_index.items()},
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist the graph to LMDB."""
        directory.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(directory / "graph.lmdb"), map_size=_LMDB_MAP_SIZE)
        try:
            with env.begin(write=True) as txn:
                # Store nodes
                node_data = {}
                for nid, node in self._nodes.items():
                    node_data[nid] = {
                        "kind": node.kind.value,
                        "label": node.label,
                        "file_path": node.file_path,
                        "metadata": node.metadata,
                    }
                txn.put(b"nodes", json.dumps(node_data).encode())

                # Store edges (outgoing adjacency list)
                edge_data = {}
                for source_id, edges in self._out_edges.items():
                    edge_data[source_id] = [
                        {"target": e.target, "type": e.type.value, "weight": e.weight, "metadata": e.metadata}
                        for e in edges
                    ]
                txn.put(b"edges", json.dumps(edge_data).encode())

                # Store indexes
                idx_data = {
                    "kind_index": {k.value: v for k, v in self._kind_index.items()},
                    "name_index": dict(self._name_index),
                    "file_to_nodes": dict(self._file_to_nodes),
                }
                txn.put(b"indexes", json.dumps(idx_data).encode())
        finally:
            env.close()

    @classmethod
    def load(cls, directory: Path) -> KnowledgeGraph | None:
        """Load a persisted graph. Returns None if no saved state found."""
        lmdb_path = directory / "graph.lmdb"
        if not lmdb_path.exists():
            return None

        graph = cls()
        env = lmdb.open(str(lmdb_path), map_size=_LMDB_MAP_SIZE)
        try:
            with env.begin() as txn:
                # Load nodes
                nodes_raw = txn.get(b"nodes")
                if nodes_raw:
                    node_data = json.loads(bytes(nodes_raw).decode())
                    for nid, data in node_data.items():
                        node = GraphNode(
                            id=nid,
                            kind=SymbolKind(data["kind"]),
                            label=data["label"],
                            file_path=data.get("file_path"),
                            metadata=data.get("metadata", {}),
                        )
                        graph._nodes[nid] = node

                # Load edges
                edges_raw = txn.get(b"edges")
                if edges_raw:
                    edge_data = json.loads(bytes(edges_raw).decode())
                    for source_id, edges_list in edge_data.items():
                        for e in edges_list:
                            edge = GraphEdge(
                                source=source_id,
                                target=e["target"],
                                type=EdgeType(e["type"]),
                                weight=e.get("weight", 1.0),
                                metadata=e.get("metadata", {}),
                            )
                            graph._out_edges[source_id].append(edge)
                            graph._in_edges[e["target"]].append(edge)

                # Load indexes
                idx_raw = txn.get(b"indexes")
                if idx_raw:
                    idx_data = json.loads(bytes(idx_raw).decode())
                    for k, v in idx_data.get("kind_index", {}).items():
                        graph._kind_index[SymbolKind(k)] = v
                    graph._name_index.update(idx_data.get("name_index", {}))
                    graph._file_to_nodes.update(idx_data.get("file_to_nodes", {}))
        finally:
            env.close()

        logger.info("Loaded graph: %d nodes, %d edges", graph.node_count, graph.edge_count)
        return graph

    def clear(self) -> None:
        """Clear all in-memory state."""
        self._nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()
        self._file_to_nodes.clear()
        self._kind_index.clear()
        self._name_index.clear()


def build_graph_from_symbols(
    symbols: list[tuple[str, list]],
    imports: list[tuple[str, list]],
    files: list[str],
) -> KnowledgeGraph:
    """Build a repository knowledge graph from parsed symbols and imports.

    This is a convenience function used during indexing.

    :param symbols: List of (file_path, symbol_list) tuples.
    :param imports: List of (file_path, import_list) tuples.
    :param files: List of all indexed file paths.
    :return: Populated KnowledgeGraph.
    """
    from vortexa.core.types import ImportInfo, SymbolInfo

    graph = KnowledgeGraph()

    # Add file nodes
    for file_path in files:
        node_id = f"file:{file_path}"
        graph.add_node(
            node_id=node_id,
            kind=SymbolKind.FILE,
            label=file_path,
            file_path=file_path,
        )

    # Add symbol nodes and CONTAINS edges
    for file_path, file_symbols in symbols:
        file_node_id = f"file:{file_path}"
        for sym in file_symbols:
            sym_node_id = _symbol_node_id(sym)
            graph.add_node(
                node_id=sym_node_id,
                kind=sym.kind,
                label=sym.name,
                file_path=sym.file_path,
                metadata={
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "docstring": sym.docstring or "",
                    "signature": sym.signature or "",
                },
            )
            # File CONTAINS symbol
            graph.add_edge(
                source=file_node_id,
                target=sym_node_id,
                edge_type=EdgeType.CONTAINS,
            )

            # Parent relationship (class contains method)
            if sym.parent:
                parent_node_id = f"{sym.kind.value}:{sym.parent}"
                if graph.has_node(parent_node_id):
                    graph.add_edge(
                        source=parent_node_id,
                        target=sym_node_id,
                        edge_type=EdgeType.CONTAINS,
                        metadata={"parent": sym.parent},
                    )

    # Add import edges
    for file_path, file_imports in imports:
        file_node_id = f"file:{file_path}"
        for imp in file_imports:
            target_module = imp.imported_module.replace(".", "/")
            # Try to find the file node that matches the import target
            imported_file_nodes = graph.find_nodes_by_kind(SymbolKind.FILE)
            for file_node in imported_file_nodes:
                if not file_node:
                    continue
                if (
                    file_node.label == target_module
                    or file_node.label.startswith(target_module)
                    or target_module in file_node.label
                ):
                    graph.add_edge(
                        source=file_node_id,
                        target=file_node.id,
                        edge_type=EdgeType.IMPORTS,
                        weight=1.0,
                        metadata={
                            "imported_module": imp.imported_module,
                            "imported_names": imp.imported_names,
                        },
                    )

    return graph


def _symbol_node_id(sym: Any) -> str:
    """Generate a unique node ID for a symbol."""
    return f"{sym.kind.value}:{sym.name}@{sym.file_path.replace('/', '.')}:{sym.start_line}"
