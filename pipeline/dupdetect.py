#!/usr/bin/env python3
"""
Code duplication audit over chunks produced by the Roslyn extractor (or ts_extract.py).

Stages
  1. filter     drop tests, generated code, boilerplate flags and tiny chunks
  2. hash       group exact copies (exactHash) and renamed copies (normalizedHash)
  3. embed      vectorize each chunk (tfidf | voyage | openai | st:<model>), cached on disk
  4. retrieve   top-k nearest neighbours per chunk above a similarity threshold
  5. verify     optional LLM pass that classifies each candidate pair (cached)
  6. cluster    connected components over kept edges, splitting oversized components
  7. summarize  optional LLM pass per top cluster: differences + suggested abstraction
  8. report     report.json (machine-readable) + report.html (self-contained)

Example
  python dupdetect.py chunks.jsonl -o audit --embedder voyage --llm anthropic
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NOISE_FLAGS = {"generated", "empty", "forwarder", "trivialCtor", "equality", "toString",
               "dispose", "propertyChanged", "diWiring"}
DEFAULT_THRESHOLDS = {"tfidf": 0.55, "voyage": 0.80, "openai": 0.80, "st": 0.85}
PROMPT_VERSION = "v1"
MAX_CODE_CHARS = 6000


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Loading and filtering
# --------------------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({e})")
    return chunks


def filter_chunks(chunks: list[dict], args) -> tuple[list[dict], Counter]:
    noise = set() if args.keep_noise else set(NOISE_FLAGS)
    if not args.include_startup:
        noise.add("startup")
    dropped = Counter()
    kept = []
    for c in chunks:
        flags = set(c.get("flags", []))
        if "test" in flags and not args.include_tests:
            dropped["test"] += 1
            continue
        hit = flags & noise
        if hit:
            dropped[sorted(hit)[0]] += 1
            continue
        # Expression-heavy code (mappers, LINQ chains) has few statements, so a chunk with
        # plenty of tokens passes even below --min-statements.
        if c["tokens"] < args.min_tokens or (c["statements"] < args.min_statements and c["tokens"] < args.min_tokens * 1.5):
            dropped["too small"] += 1
            continue
        if any(re.search(p, c["file"]) for p in args.exclude_path):
            dropped["excluded path"] += 1
            continue
        kept.append(c)
    return kept, dropped


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def embed_text(c: dict) -> str:
    return c["code"][:16000]


class EmbeddingCache:
    """hash(text) -> vector, stored as one .npz per embedder/model."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            for k, v in zip(z["keys"], z["vectors"]):
                self.data[str(k)] = v

    def save(self) -> None:
        if not self.data:
            return
        keys = np.array(list(self.data.keys()))
        vecs = np.stack(list(self.data.values())).astype(np.float32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, keys=keys, vectors=vecs)
        os.replace(tmp, self.path)


def _post_json(url: str, payload: dict, headers: dict, retries: int = 5) -> dict:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"Embedding request failed: HTTP {e.code} {e.read()[:500]!r}")
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"Embedding request failed: {e}")
    raise SystemExit("Embedding request failed after retries")


def _api_embed(texts: list[str], kind: str, model: str, batch: int) -> np.ndarray:
    out = []
    for start in range(0, len(texts), batch):
        part = texts[start:start + batch]
        if kind == "voyage":
            key = os.environ.get("VOYAGE_API_KEY") or sys.exit("VOYAGE_API_KEY is not set")
            resp = _post_json("https://api.voyageai.com/v1/embeddings",
                              {"input": part, "model": model, "input_type": "document"},
                              {"Authorization": f"Bearer {key}"})
        else:
            key = os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY is not set")
            resp = _post_json("https://api.openai.com/v1/embeddings",
                              {"input": part, "model": model},
                              {"Authorization": f"Bearer {key}"})
        rows = sorted(resp["data"], key=lambda d: d["index"])
        out.extend(r["embedding"] for r in rows)
        log(f"  embedded {min(start + batch, len(texts))}/{len(texts)}")
    return np.asarray(out, dtype=np.float32)


def embed(chunks: list[dict], args, cache_dir: Path):
    """Returns (matrix, is_sparse). Rows are L2-normalized so dot product = cosine."""
    kind = args.embedder.split(":", 1)[0]
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 3),
                              sublinear_tf=True, min_df=1, max_df=0.5 if len(chunks) > 50 else 1.0)
        X = vec.fit_transform([c["normalized"] for c in chunks])  # rows already L2-normalized
        return X.tocsr().astype(np.float32), True

    if kind == "st":
        model = args.embedder.split(":", 1)[1] if ":" in args.embedder else "all-MiniLM-L6-v2"
    elif kind == "voyage":
        model = args.embed_model or "voyage-code-3"
    elif kind == "openai":
        model = args.embed_model or "text-embedding-3-large"
    else:
        raise SystemExit(f"Unknown embedder {args.embedder!r}")

    cache = EmbeddingCache(cache_dir / f"emb-{kind}-{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}.npz")
    texts = [embed_text(c) for c in chunks]
    keys = [sha(t) for t in texts]
    missing = sorted({k: i for i, k in enumerate(keys) if k not in cache.data}.items(), key=lambda kv: kv[1])
    if missing:
        log(f"Embedding {len(missing)} chunks with {kind}:{model} ({len(keys) - len(missing)} cached)")
        todo = [texts[i] for _, i in missing]
        if kind == "st":
            from sentence_transformers import SentenceTransformer
            vecs = SentenceTransformer(model).encode(todo, batch_size=32, show_progress_bar=True,
                                                     normalize_embeddings=True)
        else:
            vecs = _api_embed(todo, kind, model, args.embed_batch)
        for (k, _), v in zip(missing, vecs):
            cache.data[k] = np.asarray(v, dtype=np.float32)
        cache.save()
    X = np.stack([cache.data[k] for k in keys]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return X, False


def top_k_neighbours(X, is_sparse: bool, k: int, block: int = 1024):
    """For each row, the k most similar other rows: (indices[n,k], sims[n,k])."""
    n = X.shape[0]
    k = min(k, max(n - 1, 1))
    idx_out = np.zeros((n, k), dtype=np.int64)
    sim_out = np.full((n, k), -1.0, dtype=np.float32)
    XT = X.T.tocsc() if is_sparse else X.T
    for start in range(0, n, block):
        stop = min(start + block, n)
        S = X[start:stop] @ XT
        S = S.toarray() if is_sparse else np.asarray(S)
        S = S.astype(np.float32, copy=False)
        S[np.arange(stop - start), np.arange(start, stop)] = -1.0  # not yourself
        part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-part_sims, axis=1)
        idx_out[start:stop] = np.take_along_axis(part, order, axis=1)
        sim_out[start:stop] = np.take_along_axis(part_sims, order, axis=1)
    return idx_out, sim_out


# --------------------------------------------------------------------------------------
# Candidate edges
# --------------------------------------------------------------------------------------

@dataclass
class Edge:
    a: int
    b: int
    sim: float
    source: str                 # exact | renamed | embedding
    verdict: str | None = None  # LLM verdict for embedding edges
    reason: str | None = None


def overlaps(a: dict, b: dict) -> bool:
    return a["file"] == b["file"] and a["startLine"] <= b["endLine"] and b["startLine"] <= a["endLine"]


def hash_edges(chunks: list[dict]) -> list[Edge]:
    edges = []
    for key, source in (("exactHash", "exact"), ("normalizedHash", "renamed")):
        groups = defaultdict(list)
        for i, c in enumerate(chunks):
            groups[c[key]].append(i)
        for members in groups.values():
            members = [m for m in members]
            if len(members) < 2:
                continue
            # star topology is enough for connectivity
            anchor = members[0]
            for m in members[1:]:
                if not overlaps(chunks[anchor], chunks[m]):
                    edges.append(Edge(anchor, m, 1.0, source))
    return edges


def embedding_edges(chunks, idx, sims, threshold: float) -> list[Edge]:
    seen = set()
    edges = []
    for i in range(len(chunks)):
        for j, s in zip(idx[i], sims[i]):
            j = int(j)
            if s < threshold or j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if overlaps(chunks[a], chunks[b]):
                continue
            if chunks[a]["normalizedHash"] == chunks[b]["normalizedHash"]:
                continue  # already covered by a hash edge
            edges.append(Edge(a, b, float(s), "embedding"))
    edges.sort(key=lambda e: -e.sim)
    return edges


# --------------------------------------------------------------------------------------
# LLM passes
# --------------------------------------------------------------------------------------

PAIR_PROMPT = """You are auditing a C# codebase for duplicated logic that could be consolidated.

Compare the two snippets below and decide whether they implement the same logic.

<snippet_a location="{loc_a}">
{code_a}
</snippet_a>

<snippet_b location="{loc_b}">
{code_b}
</snippet_b>

Verdicts:
- "equivalent": same behaviour; differences are naming, formatting or trivial style.
- "near_duplicate": same algorithm/purpose with small real differences (an extra condition, a different entity type, a different constant) that a shared helper with a parameter or generic could absorb.
- "related": similar shape or domain, but the logic differs enough that merging them would not help.
- "different": not duplicates.

Pay attention to details that change behaviour: comparison operators, null handling, off-by-one, ordering, exception types, side effects.

Reply with only a JSON object:
{{"verdict": "...", "confidence": 0.0-1.0, "reason": "one sentence naming the key difference, if any"}}"""

CLUSTER_PROMPT = """You are auditing a C# codebase for duplicated logic. These {n} code fragments were grouped as likely duplicates. The first is the most central version.

{fragments}

Assess the group and propose how to consolidate it. Be concrete and specific to this code.

Reply with only a JSON object:
{{
  "title": "short name for the duplicated logic (max 8 words)",
  "equivalence": "equivalent | mostly_equivalent | divergent",
  "summary": "1-2 sentences: what this logic does and how the copies relate",
  "differences": ["each behavioural difference between the variants, naming which variant"],
  "suggested_abstraction": "what to extract and where it should live (e.g. a generic method, extension method, shared base class, helper service)",
  "suggested_signature": "C# signature of the shared member, or empty string",
  "risk": "what could break when consolidating (one sentence)",
  "effort": "low | medium | high"
}}"""


def loc(c: dict) -> str:
    return f"{c['file']}:{c['startLine']} {c['type']}.{c['member']}"


def truncate(code: str, limit: int = MAX_CODE_CHARS) -> str:
    return code if len(code) <= limit else code[:limit] + "\n// ... truncated ..."


def parse_json_reply(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class LLM:
    def __init__(self, args, cache_dir: Path):
        import anthropic  # imported lazily so the rest works without the SDK
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (or run with --llm none)")
        self.client = anthropic.Anthropic(max_retries=6)
        self.workers = args.workers
        self.cache_path = cache_dir / "llm-cache.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                try:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["value"]
                except (json.JSONDecodeError, KeyError):
                    pass
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_path.open("a", encoding="utf-8")

    def ask(self, model: str, prompt: str, max_tokens: int) -> dict | None:
        key = sha(f"{PROMPT_VERSION}|{model}|{prompt}")
        if key in self.cache:
            return self.cache[key]
        try:
            resp = self.client.messages.create(model=model, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": prompt}])
        except Exception as e:  # keep the audit going; failures are reported
            log(f"  LLM call failed: {type(e).__name__}: {e}")
            return None
        text = "".join(getattr(b, "text", "") for b in resp.content)
        value = parse_json_reply(text)
        if value is not None:
            self.cache[key] = value
            self._cache_file.write(json.dumps({"key": key, "value": value}) + "\n")
            self._cache_file.flush()
        return value

    def map(self, model: str, prompts: list[str], max_tokens: int, label: str) -> list[dict | None]:
        results: list[dict | None] = [None] * len(prompts)
        done = 0
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.ask, model, p, max_tokens): i for i, p in enumerate(prompts)}
            for fut in cf.as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if done % 25 == 0 or done == len(prompts):
                    log(f"  {label}: {done}/{len(prompts)}")
        return results


def verify_edges(llm: LLM, chunks, edges: list[Edge], args) -> None:
    todo = edges[:args.max_llm_pairs]
    if len(edges) > len(todo):
        log(f"Verifying the {len(todo)} most similar of {len(edges)} candidate pairs (--max-llm-pairs)")
    prompts = [PAIR_PROMPT.format(loc_a=html.escape(loc(chunks[e.a]), quote=True), code_a=truncate(chunks[e.a]["code"]),
                                  loc_b=html.escape(loc(chunks[e.b]), quote=True), code_b=truncate(chunks[e.b]["code"]))
               for e in todo]
    for e, r in zip(todo, llm.map(args.pair_model, prompts, 300, "pair verification")):
        if r:
            e.verdict = str(r.get("verdict", "")).strip().lower() or None
            e.reason = r.get("reason")
    for e in edges[len(todo):]:
        e.verdict = "unverified"


def summarize_clusters(llm: LLM, chunks, clusters: list["Cluster"], args) -> None:
    todo = clusters[:args.max_llm_clusters]
    prompts = []
    for cl in todo:
        members = [cl.canonical] + [m for m in cl.members if m != cl.canonical][:args.max_variants]
        budget = max(1500, 24000 // len(members))
        frags = "\n\n".join(
            f'<fragment index="{n + 1}" location="{html.escape(loc(chunks[m]), quote=True)}">\n'
            f"{truncate(chunks[m]['code'], budget)}\n</fragment>"
            for n, m in enumerate(members))
        prompts.append(CLUSTER_PROMPT.format(n=len(members), fragments=frags))
    for cl, r in zip(todo, llm.map(args.cluster_model, prompts, 1200, "cluster summaries")):
        cl.llm = r


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def components(nodes: list[int], edges: list[Edge]) -> list[list[int]]:
    pos = {n: i for i, n in enumerate(nodes)}
    uf = UnionFind(len(nodes))
    for e in edges:
        uf.union(pos[e.a], pos[e.b])
    groups = defaultdict(list)
    for n in nodes:
        groups[uf.find(pos[n])].append(n)
    return [g for g in groups.values() if len(g) > 1]


def cluster(edges: list[Edge], X, is_sparse: bool, link_threshold: float, max_size: int) -> list[list[int]]:
    """Connected components over the kept edges, then average-linkage inside each component
    so that A~B and B~C do not chain A and C together unless the group is cohesive on
    average. Members linked by a hash edge (exact/renamed copies) always stay together."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    forced = UnionFind(X.shape[0])
    for e in edges:
        if e.source != "embedding":
            forced.union(e.a, e.b)

    def split(comp: list[int], t: float) -> list[list[int]]:
        # Collapse hash-linked members into one unit, then link the units.
        units = defaultdict(list)
        for m in comp:
            units[forced.find(m)].append(m)
        units = list(units.values())
        if len(units) == 1:
            return [sorted(comp)]
        reps = [u[0] for u in units]
        S = X[reps] @ X[reps].T
        S = S.toarray() if is_sparse else np.asarray(S)
        D = np.clip(1.0 - S, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        labels = fcluster(linkage(squareform(D, checks=False), method="average"), t=1.0 - t, criterion="distance")
        groups = defaultdict(list)
        for unit, label in zip(units, labels):
            groups[label].extend(unit)
        out = []
        for g in groups.values():
            if len(g) < 2:
                continue
            if len(g) > max_size and t < 0.99:
                out.extend(split(g, min(0.99, t + 0.03)))
            else:
                out.append(sorted(g))
        return out

    nodes = sorted({x for e in edges for x in (e.a, e.b)})
    result = []
    for comp in components(nodes, edges):
        result.extend([sorted(comp)] if len(comp) == 2 else split(comp, link_threshold))
    return result


@dataclass
class Cluster:
    members: list[int]
    tier: str
    canonical: int
    mean_sim: float
    score: float
    projects: list[str]
    verified: bool
    edges: list[Edge] = field(default_factory=list)
    llm: dict | None = None
    cid: int = 0


def ancestors(chunk_id: str, by_id: dict[str, dict]) -> set[str]:
    out = set()
    c = by_id.get(chunk_id)
    while c and c.get("parentId"):
        out.add(c["parentId"])
        c = by_id.get(c["parentId"])
    return out


def build_clusters(chunks, groups, edges, X, is_sparse, use_llm: bool, all_by_id) -> list[Cluster]:
    edge_index = defaultdict(list)
    for e in edges:
        edge_index[e.a].append(e)
        edge_index[e.b].append(e)

    clusters = []
    for g in groups:
        gs = set(g)
        inner = [e for m in g for e in edge_index[m] if e.a in gs and e.b in gs and e.a == m]
        if len({chunks[m]["exactHash"] for m in g}) == 1:
            tier = "exact"
        elif len({chunks[m]["normalizedHash"] for m in g}) == 1:
            tier = "renamed"
        else:
            tier = "near"

        sub = X[g]
        S = (sub @ sub.T)
        S = S.toarray() if is_sparse else np.asarray(S)
        canonical = g[int(np.argmax(S.sum(axis=1)))]
        off = S[~np.eye(len(g), dtype=bool)]
        mean_sim = float(off.mean()) if off.size else 1.0

        emb_edges = [e for e in inner if e.source == "embedding"]
        verified = tier != "near" or (use_llm and emb_edges and all(e.verdict in ("equivalent", "near_duplicate") for e in emb_edges))
        lines = statistics.median(chunks[m]["codeLines"] for m in g)
        projects = sorted({chunks[m]["project"] for m in g})
        types = {chunks[m]["type"] for m in g}
        tier_weight = {"exact": 1.0, "renamed": 1.0}.get(tier, 0.9 if verified else 0.6)
        spread = 1.5 if len(projects) > 1 else (1.2 if len(types) > 1 else 1.0)
        score = (len(g) - 1) * lines * tier_weight * spread
        clusters.append(Cluster(g, tier, canonical, mean_sim, score, projects, bool(verified), inner))

    # Drop clusters that are just the lambdas/local functions of an already-reported cluster.
    member_sets = [{chunks[m]["id"] for m in cl.members} for cl in clusters]
    keep = []
    for i, cl in enumerate(clusters):
        anc = [ancestors(chunks[m]["id"], all_by_id) for m in cl.members]
        subsumed = any(j != i and all(a & member_sets[j] for a in anc) for j in range(len(clusters)))
        if not subsumed:
            keep.append(cl)
    keep.sort(key=lambda c: -c.score)
    for n, cl in enumerate(keep, 1):
        cl.cid = n
    return keep


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def cluster_json(cl: Cluster, chunks) -> dict:
    def member(m):
        c = chunks[m]
        return {k: c.get(k) for k in ("id", "file", "project", "namespace", "type", "member", "kind",
                                      "startLine", "endLine", "codeLines", "hints")} | {"canonical": m == cl.canonical}
    return {
        "id": cl.cid,
        "tier": cl.tier,
        "verified": cl.verified,
        "score": round(cl.score, 1),
        "copies": len(cl.members),
        "meanSimilarity": round(cl.mean_sim, 3),
        "projects": cl.projects,
        "members": [member(m) for m in sorted(cl.members, key=lambda m: (m != cl.canonical, chunks[m]["file"]))],
        "pairs": [{"a": chunks[e.a]["id"], "b": chunks[e.b]["id"], "similarity": round(e.sim, 3),
                   "source": e.source, "verdict": e.verdict, "reason": e.reason} for e in cl.edges],
        "llm": cl.llm,
    }


def diff_html(base: str, other: str) -> str:
    rows = []
    for line in difflib.unified_diff(base.splitlines(), other.splitlines(), lineterm="", n=3):
        if line.startswith(("---", "+++")):
            continue
        cls = "hunk" if line.startswith("@@") else "add" if line.startswith("+") else "del" if line.startswith("-") else "ctx"
        rows.append(f'<span class="{cls}">{html.escape(line)}</span>')
    return "".join(rows) if rows else '<span class="ctx">(identical)</span>'


REPORT_CSS = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--muted:#6b675f;--card:#fff;--line:#e4e0d8;--accent:#3d5a80;
--exact:#2d6a4f;--renamed:#386fa4;--near:#9a6b16;--add:#e6f4ea;--addfg:#1e5e33;--del:#fbe9e7;--delfg:#8c2a1c;--code:#f4f2ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161514;--fg:#ebe8e2;--muted:#a19c92;--card:#1f1e1c;
--line:#35332f;--accent:#8fb3de;--exact:#74c69d;--renamed:#8ab8e6;--near:#e0b45c;--add:#173322;--addfg:#9be0b0;--del:#3a1d19;
--delfg:#f0a497;--code:#262522}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 4px}
.sub{color:var(--muted);margin:0 0 22px}.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 18px}
.controls input,.controls select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.controls input{flex:1;min-width:200px}
.cluster{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 14px;overflow:hidden}
.cluster>summary{list-style:none;cursor:pointer;padding:14px 16px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.cluster>summary::-webkit-details-marker{display:none}.cid{color:var(--muted);font-variant-numeric:tabular-nums}
.title{font-weight:600;flex:1;min-width:220px}.pill{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid currentColor}
.exact{color:var(--exact)}.renamed{color:var(--renamed)}.near{color:var(--near)}.muted{color:var(--muted);font-size:13px}
.body{padding:0 16px 16px;border-top:1px solid var(--line)}.body h3{font-size:14px;margin:16px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}code,pre{font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);border-radius:8px;padding:10px 12px;overflow:auto;max-height:480px;margin:6px 0}
pre span{display:block;white-space:pre;min-width:max-content}.add{background:var(--add);color:var(--addfg)}.del{background:var(--del);color:var(--delfg)}
.hunk{color:var(--muted)}.ctx{opacity:.8}ul{margin:4px 0 0;padding-left:20px}.suggest{border-left:3px solid var(--accent);padding:2px 0 2px 12px;margin:8px 0}
details.variant>summary{cursor:pointer;font-size:13px;margin-top:8px}.hidden{display:none}
td code,td span,.sub,summary code{overflow-wrap:anywhere}pre{max-width:100%}.cluster{min-width:0}
@media (max-width:640px){td:nth-child(3),th:nth-child(3){display:none}}
"""

REPORT_JS = """
const q=document.getElementById('q'),tier=document.getElementById('tier'),proj=document.getElementById('proj');
function apply(){const t=q.value.toLowerCase();for(const el of document.querySelectorAll('.cluster')){
const ok=(!t||el.dataset.text.includes(t))&&(!tier.value||el.dataset.tier===tier.value)&&(!proj.value||el.dataset.projects.split('|').includes(proj.value));
el.classList.toggle('hidden',!ok)}}
for(const el of [q,tier,proj])el.addEventListener('input',apply);
"""


def render_html(report: dict, chunks_by_id: dict[str, dict]) -> str:
    e = html.escape
    s = report["stats"]
    projects = sorted({p for cl in report["clusters"] for p in cl["projects"]})
    cards = []
    for cl in report["clusters"]:
        llm = cl["llm"] or {}
        canonical = next(m for m in cl["members"] if m["canonical"])
        title = llm.get("title") or f"{canonical['type']}.{canonical['member']}"
        tier_label = {"exact": "exact copy", "renamed": "renamed copy", "near": "near duplicate"}[cl["tier"]]
        if cl["tier"] == "near" and not cl["verified"]:
            tier_label += " (unverified)"
        rows = "".join(
            f"<tr><td>{'★ ' if m['canonical'] else ''}<code>{e(m['type'])}.{e(m['member'])}</code></td>"
            f"<td><code>{e(m['file'])}:{m['startLine']}</code></td><td>{e(m['project'])}</td><td>{m['codeLines']}</td></tr>"
            for m in cl["members"])
        parts = [f"<h3>Locations</h3><table><tr><th>Member</th><th>File</th><th>Project</th><th>Lines</th></tr>{rows}</table>"]
        if llm:
            diffs = "".join(f"<li>{e(str(d))}</li>" for d in llm.get("differences") or [])
            parts.append("<h3>Assessment</h3>"
                         f"<p>{e(llm.get('summary', ''))}</p>"
                         + (f"<p class='muted'>Differences:</p><ul>{diffs}</ul>" if diffs else "")
                         + f"<div class='suggest'><b>Suggested:</b> {e(llm.get('suggested_abstraction', ''))}"
                         + (f"<pre><span>{e(llm['suggested_signature'])}</span></pre>" if llm.get("suggested_signature") else "")
                         + f"<span class='muted'>Risk: {e(llm.get('risk', ''))} · Effort: {e(llm.get('effort', '?'))} · "
                           f"Equivalence: {e(llm.get('equivalence', '?'))}</span></div>")
        reasons = [p for p in cl["pairs"] if p.get("reason")]
        if reasons and not llm:
            parts.append("<h3>Pair notes</h3><ul>" + "".join(
                f"<li><span class='muted'>{e(p['verdict'] or '')}</span> {e(p['reason'])}</li>" for p in reasons[:6]) + "</ul>")
        base_code = chunks_by_id[canonical["id"]]["code"]
        parts.append(f"<h3>Canonical version</h3><pre>{''.join(f'<span>{e(l)}</span>' for l in base_code.splitlines())}</pre>")
        for m in cl["members"]:
            if m["canonical"]:
                continue
            other = chunks_by_id[m["id"]]["code"]
            parts.append(f"<details class='variant'><summary>Diff vs <code>{e(m['type'])}.{e(m['member'])}</code> "
                         f"({e(m['file'])}:{m['startLine']})</summary><pre>{diff_html(base_code, other)}</pre></details>")
        search_text = " ".join([title] + [f"{m['file']} {m['type']} {m['member']}" for m in cl["members"]]).lower()
        cards.append(
            f"<details class='cluster' data-tier='{cl['tier']}' data-projects='{e('|'.join(cl['projects']))}' data-text='{e(search_text)}'>"
            f"<summary><span class='cid'>#{cl['id']}</span><span class='title'>{e(title)}</span>"
            f"<span class='pill {cl['tier']}'>{e(tier_label)}</span>"
            f"<span class='muted'>{cl['copies']} copies · {len(cl['projects'])} project{'s' if len(cl['projects']) != 1 else ''} · "
            f"sim {cl['meanSimilarity']:.2f} · score {cl['score']:.0f}</span></summary>"
            f"<div class='body'>{''.join(parts)}</div></details>")

    lookalike_html = ""
    if report.get("lookalikes"):
        items = "".join(
            f"<tr><td><code>{e(l['aName'])}</code><br><span class='muted'>{e(l['a'])}</span></td>"
            f"<td><code>{e(l['bName'])}</code><br><span class='muted'>{e(l['b'])}</span></td>"
            f"<td>{l['similarity']:.2f}</td><td>{e(l.get('reason') or '')}</td></tr>"
            for l in report["lookalikes"][:200])
        lookalike_html = ("<h2>Look-alikes not merged</h2><p class='sub'>Similar code the verifier judged different enough "
                          "to keep apart. Small behavioural differences here are sometimes bugs.</p>"
                          f"<table><tr><th>A</th><th>B</th><th>Sim</th><th>Difference</th></tr>{items}</table>")
    tier_counts = Counter(cl["tier"] for cl in report["clusters"])
    stat = lambda v, label: f"<div class='stat'><b>{v}</b><span>{label}</span></div>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Duplication Audit</title>
<style>{REPORT_CSS}</style></head><body><main>
<h1>Duplication audit</h1>
<p class="sub">{e(s['source'])} · {s['chunksTotal']} chunks extracted, {s['chunksAnalyzed']} analyzed · embedder {e(s['embedder'])} · threshold {s['threshold']} · generated {e(s['generatedAt'])}</p>
<div class="stats">{stat(len(report['clusters']), 'clusters')}{stat(s['duplicatedLines'], 'duplicated lines (approx.)')}
{stat(tier_counts.get('exact', 0), 'exact')}{stat(tier_counts.get('renamed', 0), 'renamed')}{stat(tier_counts.get('near', 0), 'near')}</div>
<div class="controls"><input id="q" placeholder="Filter by file, type or member…">
<select id="tier"><option value="">All tiers</option><option value="exact">Exact</option><option value="renamed">Renamed</option><option value="near">Near</option></select>
<select id="proj"><option value="">All projects</option>{''.join(f'<option>{e(p)}</option>' for p in projects)}</select></div>
{''.join(cards) or '<p>No duplicate clusters found.</p>'}
{lookalike_html}
</main><script>{REPORT_JS}</script></body></html>"""


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("chunks", type=Path, help="chunks.jsonl from the extractor")
    p.add_argument("-o", "--out", type=Path, default=Path("dup-audit"), help="output directory")
    g = p.add_argument_group("filtering")
    g.add_argument("--min-tokens", type=int, default=40, help="minimum normalized tokens per chunk (default 40)")
    g.add_argument("--min-statements", type=int, default=3, help="minimum statements per chunk unless it has 1.5x --min-tokens (default 3)")
    g.add_argument("--include-tests", action="store_true", help="include chunks flagged as test code")
    g.add_argument("--include-startup", action="store_true", help="include Program.cs / Startup.cs")
    g.add_argument("--keep-noise", action="store_true", help="keep boilerplate flags: " + ", ".join(sorted(NOISE_FLAGS)))
    g.add_argument("--exclude-path", action="append", default=[], metavar="REGEX", help="drop files matching regex (repeatable)")
    g = p.add_argument_group("retrieval")
    g.add_argument("--embedder", default="tfidf", help="tfidf | voyage | openai | st:<sentence-transformers model> (default tfidf)")
    g.add_argument("--embed-model", help="model name for voyage/openai (defaults: voyage-code-3, text-embedding-3-large)")
    g.add_argument("--embed-batch", type=int, default=64)
    g.add_argument("--threshold", type=float, help="cosine similarity cut-off (default depends on embedder)")
    g.add_argument("--top-k", type=int, default=8, help="neighbours retrieved per chunk (default 8)")
    g.add_argument("--link-threshold", type=float,
                   help="average similarity a cluster must keep (default: threshold with --llm, "
                        "threshold + 15%% of the remaining gap to 1.0 without)")
    g.add_argument("--max-cluster", type=int, default=25, help="split clusters larger than this (default 25)")
    g = p.add_argument_group("LLM")
    g.add_argument("--llm", choices=["none", "anthropic"], default="none")
    g.add_argument("--pair-model", default="claude-haiku-5-5", help="model for pair verification")
    g.add_argument("--cluster-model", default="claude-sonnet-5-5", help="model for cluster summaries")
    g.add_argument("--max-llm-pairs", type=int, default=2000)
    g.add_argument("--max-llm-clusters", type=int, default=60)
    g.add_argument("--max-variants", type=int, default=5, help="variants shown to the LLM per cluster besides the canonical one")
    g.add_argument("--workers", type=int, default=8, help="concurrent LLM requests")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / ".cache"
    kind = args.embedder.split(":", 1)[0]
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS.get(kind, 0.8)

    all_chunks = load_chunks(args.chunks)
    all_by_id = {c["id"]: c for c in all_chunks}
    chunks, dropped = filter_chunks(all_chunks, args)
    log(f"Loaded {len(all_chunks)} chunks; analyzing {len(chunks)}. Dropped: "
        + (", ".join(f"{k}={v}" for k, v in dropped.most_common()) or "none"))
    if len(chunks) < 2:
        log("Not enough chunks to compare.")
        return 1

    edges = hash_edges(chunks)
    log(f"Hash pass: {sum(e.source == 'exact' for e in edges)} exact + {sum(e.source == 'renamed' for e in edges)} renamed links")

    X, is_sparse = embed(chunks, args, cache_dir)
    idx, sims = top_k_neighbours(X, is_sparse, args.top_k)
    top1 = sims[:, 0]
    qs = np.quantile(top1, [0.5, 0.9, 0.99])
    log(f"Nearest-neighbour similarity: median {qs[0]:.3f}, p90 {qs[1]:.3f}, p99 {qs[2]:.3f} "
        f"(threshold {threshold}; {int((top1 >= threshold).sum())} chunks have a neighbour above it)")

    cand = embedding_edges(chunks, idx, sims, threshold)
    log(f"Embedding pass: {len(cand)} candidate pairs")

    llm = None
    if args.llm == "anthropic" and (cand or edges):
        llm = LLM(args, cache_dir)
    if llm and cand:
        verify_edges(llm, chunks, cand, args)
        verdicts = Counter(e.verdict or "error" for e in cand)
        log("Verdicts: " + ", ".join(f"{k}={v}" for k, v in verdicts.most_common()))
        kept_emb = [e for e in cand if e.verdict in ("equivalent", "near_duplicate", "unverified")]
        lookalikes = [e for e in cand if e.verdict == "related"]
    else:
        kept_emb = cand
        lookalikes = []

    all_edges = edges + kept_emb
    link = args.link_threshold if args.link_threshold is not None else (
        threshold if llm else threshold + 0.15 * (1 - threshold))
    groups = cluster(all_edges, X, is_sparse, link, args.max_cluster)
    clusters = build_clusters(chunks, groups, all_edges, X, is_sparse, llm is not None, all_by_id)
    log(f"Clusters: {len(clusters)}")

    if llm and clusters:
        summarize_clusters(llm, chunks, clusters, args)

    duplicated_lines = int(sum(sum(chunks[m]["codeLines"] for m in cl.members if m != cl.canonical) for cl in clusters))
    report = {
        "stats": {
            "source": str(args.chunks),
            "generatedAt": time.strftime("%Y-%m-%d %H:%M"),
            "chunksTotal": len(all_chunks),
            "chunksAnalyzed": len(chunks),
            "dropped": dict(dropped),
            "embedder": args.embedder if kind != "voyage" and kind != "openai" else f"{kind}:{args.embed_model or ''}".rstrip(":"),
            "threshold": threshold,
            "linkThreshold": round(link, 3),
            "candidatePairs": len(cand),
            "llm": args.llm,
            "duplicatedLines": duplicated_lines,
            "seconds": round(time.time() - t0, 1),
        },
        "clusters": [cluster_json(cl, chunks) for cl in clusters],
        # Similar-looking code the LLM judged not worth merging. Worth a glance: small
        # behavioural differences between look-alikes are sometimes bugs.
        "lookalikes": [{"a": chunks[e.a]["id"], "b": chunks[e.b]["id"],
                        "aName": f"{chunks[e.a]['type']}.{chunks[e.a]['member']}",
                        "bName": f"{chunks[e.b]['type']}.{chunks[e.b]['member']}",
                        "similarity": round(e.sim, 3), "reason": e.reason} for e in lookalikes],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "report.html").write_text(render_html(report, all_by_id), encoding="utf-8")
    log(f"Wrote {args.out / 'report.html'} and {args.out / 'report.json'} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
