"""Test VortexA v2 Context Engine end-to-end.

Usage:
    cd /home/vortex/Desktop/CODEBASE/Projects/OEvortex/vortexa
    .venv/bin/python -m vortexa.core.v2_test

Or:
    PYTHONPATH=src python vortexa/core/v2_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add vortexa to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vortexa.core.context_engine import VortexContextEngine
from vortexa.core.graph import RepoGraphBuilder
from vortexa.core.v3_embedder import VortexEmbedderV3


# Test queries from the Webscout code-search benchmark
TEST_QUERIES = [
    ("openai chat completion streaming response handler", ["model_fetcher.py"]),
    ("anthropic claude conversation messages api", ["model_fetcher.py"]),
    ("google gemini generative ai client", ["model_fetcher.py"]),
    ("perplexity search web results", ["Perplexity.py"]),
    ("elevenlabs text to speech voice synthesis", ["elevenlabs.py"]),
    ("proxy manager network rotation", ["ProxyScrapeAPI", "ProxyNova", "proxyscrape", "proxy_nova"]),
    ("duckduckgo search engine scraper", ["duckduckgo"]),
    ("bing search engine scrape results", ["bing"]),
    ("yahoo search engine", ["yahoo"]),
    ("scout zeroart ascii art", ["zeroart"]),
    ("swiftcli swift command shell", ["swiftcli"]),
    ("aiutel aibase aisearch monica webpilot", ["AIutel", "AIbase", "AISEARCH", "monica_search", "webpilotai_search"]),
    ("aibase brave iask monica webpilot", ["BraveSearch", "iask_search", "monica_search", "webpilotai_search"]),
    ("extra embedding ai4chat search engines", ["Extra", "embedding", "search/engines"]),
    ("yt-dlp youtube video downloader", ["Ytb"]),
    ("spotify music streaming api", ["Spotify"]),
    ("kocoban telegram bot downloader", ["kocoban", "Telegram"]),
    ("weather api openweathermap forecast", ["OpenWeatherMap"]),
    ("arxiv paper search academic", ["arxiv"]),
    ("github code repository search", ["github"]),
    ("coinbase crypto exchange api", ["Coinbase"]),
    ("wolfram alpha computational engine", ["WolframAlpha"]),
    ("stackoverflow programming qa", ["StackOverflow"]),
    ("pollinations ai image generation", ["pollinations"]),
    ("tiktok video downloader", ["TikTok"]),
    ("reddit rss feed parser", ["reddit"]),
    ("scrapingant web scraper", ["ScrapingAnt", "scrapingant"]),
    ("phind code search assistant", ["phind"]),
    ("komo ai search", ["komo"]),
    ("hackernews ycombinator news", ["HackerNews", "YCombinator"]),
    ("you search engine", ["YouSearch"]),
    ("mojeek privacy search", ["mojeek"]),
    ("alexa internet rank", ["AlexaInternet"]),
    ("qwant search engine", ["Qwant"]),
    ("unsource search", ["unsource"]),
    ("searx metasearch", ["Searx", "searx"]),
    ("baidu search chinese", ["baidu"]),
    ("yep search engine", ["Yep", "YepSearch"]),
    ("lingvana translate", ["Lingvanex", "translate"]),
    ("toxicity detection huggingface", ["toxic"]),
    ("jailbreak prompts", ["jailbreak"]),
    ("deepseek r1 reasoning model", ["deepseek"]),
    ("openrouter ai router", ["openrouter"]),
    ("yandex search", ["yandex"]),
    ("kagi search engine", ["kagi"]),
    ("thinkany ai", ["ThinkAnyAI", "thinkany"]),
    ("textcortex ai writer", ["TextCortex", "textcortex"]),
    ("cerebras inference", ["Cerebras"]),
    ("brave search", ["BraveSearch"]),
    ("consensus research search", ["Consensus"]),
    ("ask for search", ["ask"]),
    ("translate french german spanish", ["Translate"]),
]


WEBSCOUT = Path("/home/vortex/Desktop/CODEBASE/Projects/OEvortex/Webscout")


def main():
    if not WEBSCOUT.exists():
        print(f"Webscout not found at {WEBSCOUT}")
        return

    # 1. Collect files
    print("Collecting files...")
    files = {}
    for p in WEBSCOUT.rglob("*.py"):
        sp = str(p)
        if any(s in sp for s in ("__pycache__", "/dist/", "/.venv/", "node_modules")):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip() or len(text) > 100_000:
            continue
        rel = str(p.relative_to(WEBSCOUT))
        files[rel] = text
    print(f"  {len(files)} files")

    # 2. Build graph
    print("Building graph...")
    t0 = time.perf_counter()
    gb = RepoGraphBuilder()
    graph = gb.build(files)
    print(f"  {graph.stats()} in {time.perf_counter()-t0:.1f}s")

    # 3. Build adjacency and imports
    adjacency = {}
    imports_map = {}
    for nid, edges in graph._out.items():
        path = nid.split("file:")[-1] if nid.startswith("file:") else None
        if not path:
            continue
        for e in edges:
            if not e.dst.startswith("file:"):
                continue
            dst = e.dst.split("file:")[-1]
            adjacency.setdefault(path, set()).add(dst)
            if e.kind == "IMPORTS":
                imports_map.setdefault(path, set()).add(dst)

    # 4. Build file_idf (per-file term frequencies)
    print("Building file_idf...")
    file_idf = {}
    df = {}  # doc freq
    for path, content in files.items():
        terms = set()
        for t in __import__("re").findall(r"[a-z0-9_]+", content.lower()):
            terms.add(t)
        for t in terms:
            df[t] = df.get(t, 0) + 1
    n_files = len(files)
    for path, content in files.items():
        terms = {}
        for t in __import__("re").findall(r"[a-z0-9_]+", content.lower()):
            terms[t] = terms.get(t, 0) + 1
        idf = {}
        for t, tf in terms.items():
            # Log-scaled idf
            doc_freq = df.get(t, 1)
            idf[t] = max(0.0, np.log(n_files / doc_freq))
        file_idf[path] = idf

    # 5. Chunk and embed
    print("Chunking and embedding...")
    t0 = time.perf_counter()
    chunks = []
    from vortexa.core.chunking import chunk_source
    from vortexa.core.language import detect_language
    for path, content in files.items():
        language = detect_language(path)
        try:
            cs = chunk_source(content, path, language)
            for c in cs:
                # c has .text, .start_line, .end_line, etc.
                text = getattr(c, 'text', None) or getattr(c, 'content', None) or str(c)
                sl = getattr(c, 'start_line', 0)
                el = getattr(c, 'end_line', 0)
                chunks.append({
                    "path": path,
                    "content": text,
                    "start": sl,
                    "end": el,
                    "chunk_id": f"{path}:{sl}",
                })
        except Exception as e:
            # Fallback: line-level chunking
            lines = content.splitlines()
            for i in range(0, len(lines), 80):
                end = min(i + 80, len(lines))
                chunks.append({
                    "path": path,
                    "content": "\n".join(lines[i:end]),
                    "start": i + 1,
                    "end": end,
                    "chunk_id": f"{path}:{i+1}",
                })
    print(f"  {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")

    # 6. Load v3 embedder and encode
    print("Loading v3 embedder...")
    t0 = time.perf_counter()
    embedder = VortexEmbedderV3()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Fit SIF+PC on a sample
    print("Fitting SIF+PC...")
    t0 = time.perf_counter()
    sample = [c["content"][:400] for c in chunks[:1000]]
    embedder.fit_corpus(sample)
    print(f"  fitted in {time.perf_counter()-t0:.1f}s")

    # Encode all chunks
    print("Encoding chunks...")
    t0 = time.perf_counter()
    chunk_texts = [c["content"] for c in chunks]
    # Batch encode
    batch_size = 64
    all_embs = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        embs = embedder.embed_batch(batch)
        all_embs.append(embs)
    chunk_embeddings = np.vstack(all_embs)
    print(f"  {chunk_embeddings.shape} in {time.perf_counter()-t0:.1f}s")

    # 7. Build V2 engine
    engine = VortexContextEngine(
        embedder=embedder,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        graph=graph,
        file_imports=imports_map,
        file_adjacency=adjacency,
        file_idf=file_idf,
    )

    # 8. Run benchmark
    print()
    print("=" * 70)
    print("V2 CONTEXT ENGINE BENCHMARK (file-level R@1, R@5)")
    print("=" * 70)
    hits_at_1 = 0
    hits_at_5 = 0
    n = 0
    for query, expected in TEST_QUERIES:
        pack = engine.search(query, top_k_dense=100, top_k_final=10)
        all_paths = [cf.path for cf in pack.primary] + [cf.path for cf in pack.related]
        rank = None
        for r, path in enumerate(all_paths, 1):
            if any(exp.lower() in path.lower() for exp in expected):
                rank = r
                break
        if rank is not None:
            if rank <= 1: hits_at_1 += 1
            if rank <= 5: hits_at_5 += 1
        n += 1
        mark = '✓' if rank == 1 else ('·' if rank is not None and rank <= 5 else '✗')
        rank_str = f"R@{rank}" if rank else "miss"
        top1 = pack.primary[0].path if pack.primary else "none"
        print(f"  {mark} {rank_str:6s} {query[:40]:40s} → {top1}")
    print()
    print(f"R@1: {hits_at_1}/{n} = {hits_at_1/n:.2%}")
    print(f"R@5: {hits_at_5}/{n} = {hits_at_5/n:.2%}")


if __name__ == "__main__":
    main()
