"""V2 with line-level chunking to match v1's prior performance."""
import sys
import time
import re
from pathlib import Path
from collections import Counter
from typing import List

import numpy as np

sys.path.insert(0, '/home/vortex/Desktop/CODEBASE/Projects/OEvortex/vortexa/src')
sys.path.insert(0, '/home/vortex/Desktop/lf1bit')

from vortexa.core.context_engine import VortexContextEngine
from vortexa.core.graph import RepoGraphBuilder
from vortexa.core.v3_embedder import VortexEmbedderV3
from vortexa.core.vortex_score import VortexScoreWeights


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


def chunk_files_line_level(corpus_dir, chunk_size=80, chunk_overlap=20, max_chunks=10000):
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for p in sorted(corpus_dir.rglob("*.py")):
        sp = str(p)
        if any(s in sp for s in ("__pycache__", "/dist/", "/.venv/", "node_modules", "/tests/")):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        rel = str(p.relative_to(corpus_dir))
        lines = text.splitlines()
        for i in range(0, len(lines), step):
            end_idx = min(i + chunk_size, len(lines))
            content = "\n".join(lines[i:end_idx])
            if content.strip():
                chunks.append({"path": rel, "content": content, "start": i + 1, "end": end_idx, "chunk_id": f"{rel}:{i+1}"})
            if end_idx >= len(lines):
                break
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def main():
    print("Building V2 with LINE-LEVEL chunking (matches v1's prior benchmark setup)...")
    print()

    # Collect files
    files = {}
    for p in WEBSCOUT.rglob("*.py"):
        sp = str(p)
        if any(s in sp for s in ("__pycache__", "/dist/", "/.venv/", "node_modules", "/tests/")):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip() or len(text) > 100_000:
            continue
        rel = str(p.relative_to(WEBSCOUT))
        files[rel] = text
    print(f"Files: {len(files)}")

    # Build graph
    print("Building graph...")
    gb = RepoGraphBuilder()
    graph = gb.build(files)
    print(f"  {graph.stats()}")

    # Adjacency and imports
    adjacency = {}
    imports_map = {}
    for nid, edges in graph._out.items():
        path = nid.split("file:")[-1] if nid.startswith("file:") else None
        if not path: continue
        for e in edges:
            if not e.dst.startswith("file:"): continue
            dst = e.dst.split("file:")[-1]
            adjacency.setdefault(path, set()).add(dst)
            if e.kind == "IMPORTS":
                imports_map.setdefault(path, set()).add(dst)

    # IDF
    file_idf = {}
    df = {}
    for path, content in files.items():
        terms = set(re.findall(r"[a-z0-9_]+", content.lower()))
        for t in terms:
            df[t] = df.get(t, 0) + 1
    n_files = len(files)
    for path, content in files.items():
        terms = {}
        for t in re.findall(r"[a-z0-9_]+", content.lower()):
            terms[t] = terms.get(t, 0) + 1
        idf = {}
        for t, tf in terms.items():
            doc_freq = df.get(t, 1)
            idf[t] = max(0.0, np.log(n_files / doc_freq))
        file_idf[path] = idf

    # Line-level chunks
    chunks = chunk_files_line_level(WEBSCOUT, chunk_size=80, chunk_overlap=20)
    print(f"Chunks: {len(chunks)}")

    # v3 embedder
    embedder = VortexEmbedderV3()
    sample = [c["content"][:400] for c in chunks[:1000]]
    embedder.fit_corpus(sample)

    print("Encoding chunks...")
    all_texts = [c["content"] for c in chunks]
    batch_size = 64
    all_embs = []
    for i in range(0, len(chunks), batch_size):
        all_embs.append(embedder.embed_batch([c["content"] for c in chunks[i:i+batch_size]]))
    chunk_embeddings = np.vstack(all_embs)
    print(f"  {chunk_embeddings.shape}")

    # Test 3 weight profiles
    print()
    print("=" * 70)
    print("V2 + LINE-LEVEL + WEIGHT SWEEP")
    print("=" * 70)
    for weights, name in [
        (VortexScoreWeights(embedding=1.0, filename=0, path=0, symbol=0, graph=0, import_rel=0, idf=0), "emb-only"),
        (VortexScoreWeights(embedding=0.3, filename=0.3, path=0.1, symbol=0.2, graph=0.05, import_rel=0, idf=0.05), "filename-heavy"),
        (VortexScoreWeights(), "default"),
    ]:
        engine = VortexContextEngine(
            embedder=embedder, chunks=chunks, chunk_embeddings=chunk_embeddings,
            graph=graph, file_imports=imports_map, file_adjacency=adjacency,
            file_idf=file_idf, weights=weights,
        )
        # Evaluate
        hits_1 = hits_5 = 0
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
                if rank <= 1: hits_1 += 1
                if rank <= 5: hits_5 += 1
            n += 1
        r1 = hits_1 / n
        r5 = hits_5 / n
        print(f"  {name:20s}: R@1={r1:.3f}  R@5={r5:.3f}")


if __name__ == "__main__":
    main()
