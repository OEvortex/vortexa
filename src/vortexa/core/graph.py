"""Repository Knowledge Graph for VortexA v2.

Extracts symbols, imports, calls, and references from source code.
Builds a graph that supports neighbor queries, expansion, and structural
retrieval.

Graph nodes:
    File
    Class
    Function
    Method
    Symbol (variable, constant)

Graph edges:
    IMPORTS  : file -> file
    IMPORTS_FROM : file -> symbol
    DEFINES  : file -> symbol
    CALLS    : symbol -> symbol
    USES     : symbol -> symbol
    EXTENDS  : class -> class
    REFERENCES : file -> file
"""
from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the repo graph."""
    id: str  # "file:path/to/file.py" or "sym:path/to/file.py::func"
    kind: str  # "file", "function", "class", "method", "symbol"
    name: str
    path: str  # file path (relative)
    line: int = 0
    docstring: str = ""


@dataclass
class GraphEdge:
    """An edge in the repo graph."""
    src: str  # source node id
    dst: str  # destination node id
    kind: str  # IMPORTS, CALLS, EXTENDS, etc.
    weight: float = 1.0


@dataclass
class RepoGraph:
    """In-memory repository knowledge graph.

    Lightweight: just dicts and sets. No database.
    Supports neighbor lookup, expansion, and simple scoring.
    """
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    # Adjacency lists
    _out: Dict[str, List[GraphEdge]] = field(default_factory=lambda: defaultdict(list))
    _in: Dict[str, List[GraphEdge]] = field(default_factory=lambda: defaultdict(list))
    # File -> set of symbol ids defined in that file
    _file_symbols: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # Symbol name -> set of fully-qualified ids (for resolution)
    _name_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        self._name_index[node.name].add(node.id)
        if node.kind != "file":
            self._file_symbols[node.path].add(node.id)

    def add_edge(self, src: str, dst: str, kind: str, weight: float = 1.0) -> None:
        if src not in self.nodes or dst not in self.nodes:
            return
        edge = GraphEdge(src=src, dst=dst, kind=kind, weight=weight)
        self.edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)

    def neighbors(self, node_id: str, kind: Optional[str] = None, direction: str = "out") -> List[str]:
        if node_id not in self.nodes:
            return []
        edges = self._out.get(node_id, []) if direction == "out" else self._in.get(node_id, [])
        if kind is not None:
            return [e.dst if direction == "out" else e.src for e in edges if e.kind == kind]
        return [e.dst if direction == "out" else e.src for e in edges]

    def expand(self, seed_ids: List[str], max_hops: int = 2, max_size: int = 100) -> List[Tuple[str, int]]:
        """BFS from seed nodes, return (node_id, hop_count) pairs."""
        visited: Set[str] = set()
        frontier: List[Tuple[str, int]] = [(s, 0) for s in seed_ids if s in self.nodes]
        for sid, _ in frontier:
            visited.add(sid)
        for hop in range(1, max_hops + 1):
            new_frontier: List[Tuple[str, int]] = []
            for sid, _ in frontier:
                for nb in self.neighbors(sid):
                    if nb not in visited:
                        visited.add(nb)
                        new_frontier.append((nb, hop))
            frontier = new_frontier
            if len(visited) >= max_size:
                break
        # Sort by hop count
        all_nodes = [(s, 0) for s in seed_ids if s in self.nodes]
        all_nodes.extend(frontier)
        return all_nodes[:max_size]

    def file_symbols(self, path: str) -> Set[str]:
        return self._file_symbols.get(path, set())

    def resolve_name(self, name: str) -> List[str]:
        """Resolve a bare name to a list of fully-qualified node ids."""
        return list(self._name_index.get(name, set()))

    def stats(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_files": sum(1 for n in self.nodes.values() if n.kind == "file"),
            "n_functions": sum(1 for n in self.nodes.values() if n.kind in ("function", "method")),
            "n_classes": sum(1 for n in self.nodes.values() if n.kind == "class"),
        }


class RepoGraphBuilder:
    """Build a RepoGraph from a Python repository."""

    def __init__(self) -> None:
        self.graph = RepoGraph()

    def add_file(self, path: str, content: str) -> None:
        """Parse a file and add its symbols/edges to the graph."""
        file_id = f"file:{path}"
        if file_id not in self.graph.nodes:
            self.graph.add_node(GraphNode(
                id=file_id, kind="file", name=path.split("/")[-1], path=path,
            ))
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            return
        # Build symbol definitions in this file
        # First pass: collect class and function names
        scope_stack: List[str] = []  # for nested classes/methods
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fq_name = self._make_fq(path, scope_stack, node.name)
                docstring = ast.get_docstring(node) or ""
                try:
                    line = node.lineno
                except AttributeError:
                    line = 0
                self.graph.add_node(GraphNode(
                    id=fq_name, kind="function", name=node.name,
                    path=path, line=line, docstring=docstring,
                ))
                self.graph.add_edge(file_id, fq_name, "DEFINES")
            elif isinstance(node, ast.ClassDef):
                fq_name = self._make_fq(path, scope_stack, node.name)
                docstring = ast.get_docstring(node) or ""
                try:
                    line = node.lineno
                except AttributeError:
                    line = 0
                self.graph.add_node(GraphNode(
                    id=fq_name, kind="class", name=node.name,
                    path=path, line=line, docstring=docstring,
                ))
                self.graph.add_edge(file_id, fq_name, "DEFINES")
        # Second pass: imports + references
        self._add_imports(tree, path, file_id)
        self._add_references(tree, path, file_id)

    def _make_fq(self, path: str, scope: List[str], name: str) -> str:
        """Build a fully-qualified symbol id."""
        if scope:
            return f"sym:{path}::{'::'.join(scope)}::{name}"
        return f"sym:{path}::{name}"

    def _add_imports(self, tree: ast.AST, path: str, file_id: str) -> None:
        """Add import edges."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Edge: this file -> imported module (file level)
                    target = alias.name.split(".")[0]
                    self._link_file(file_id, target, "IMPORTS")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                # Edge: this file -> imported module
                target = node.module.split(".")[0]
                self._link_file(file_id, target, "IMPORTS")
                # Edge: each imported name -> this file
                for alias in node.names:
                    imported_name = alias.name
                    # Find the fully-qualified id in the imported module
                    target_ids = self.graph.resolve_name(imported_name)
                    for tid in target_ids:
                        if tid != file_id:
                            self.graph.add_edge(file_id, tid, "IMPORTS_FROM", weight=0.5)

    def _link_file(self, src_file_id: str, target_module: str, kind: str) -> None:
        """Try to link to a known file node, otherwise create a stub."""
        # Try common patterns
        target_path = target_module + ".py"
        target_id = f"file:{target_path}"
        if target_id in self.graph.nodes:
            self.graph.add_edge(src_file_id, target_id, kind)
            return
        # Also try as a submodule
        for nid in self.graph.nodes:
            if nid.endswith(f"/{target_path}") or nid.endswith(target_path):
                self.graph.add_edge(src_file_id, nid, kind)
                return

    def _add_references(self, tree: ast.AST, path: str, file_id: str) -> None:
        """Find references to known symbols (best-effort)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Try to resolve the call
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    continue
                # Find candidates
                candidates = self.graph.resolve_name(name)
                for cid in candidates[:3]:  # limit candidates
                    self.graph.add_edge(file_id, cid, "REFERENCES", weight=0.3)

    def build(self, files: Dict[str, str]) -> RepoGraph:
        """Build the graph from a {path: content} dict."""
        for path, content in files.items():
            self.add_file(path, content)
        return self.graph
