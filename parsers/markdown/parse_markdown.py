#!/usr/bin/env python3
"""Markdown -> tree-structured JSON for llama-treedriver  (domain 3: Markdown documents).

Idea
  * Every heading section is a node of a tree (document -> h1 -> h2 -> h3 ...).
  * Every node gets its own sequence id.  A token of node N carries the ids of N and of all
    nodes below N (seq_ids[i][0] is N itself).  So a section is visible to its own
    descendants but a section never sees its siblings or their children.
  * Positions: a node starts right after its parent's own tokens, so sibling sections
    share the same start position.  Positions are consecutive within every sequence id.
  * Tiny subtrees (fewer than --min-tokens tokens) are merged into their parent's sequence.
    If more than 63 sequences remain, the threshold is raised automatically
    (llama.cpp context is created with n_seq_max = 64).

Output (JSON): parallel arrays tokens / positions / seq_ids / node_type / parent  (+ is_markup),
plus "nodes" and "meta" for tests and statistics (the C++ driver ignores them).

Usage
  python3 parse_markdown.py doc.md -o doc.json --model models/X.gguf --tokenizer build/bin/llama-tokenize
  python3 parse_markdown.py doc.md -o doc.json --fake-tokenizer          # offline test, no model needed
"""
import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

from markdown_it import MarkdownIt

MAX_SEQ = 64          # must equal n_seq_max in treedriver.cpp
FAKE_BOS = 128000


# --------------------------------------------------------------------------- tokenizer
class Tokenizer:
    """Wraps llama-tokenize (exact token ids of your model) or a fake word tokenizer for offline tests."""

    def __init__(self, tok_bin=None, model=None, fake=False):
        self.tok_bin, self.model, self.fake = tok_bin, model, fake
        self.cache = {}

    def _run(self, text, bos):
        if text == "":
            return []
        if self.fake:
            ids = [zlib.crc32(w.encode()) % 100000 + 1000 for w in re.findall(r"\w+|\W", text)]
            return ([FAKE_BOS] if bos else []) + ids
        cmd = [self.tok_bin, "-m", self.model, "--ids", "--log-disable", "-p", text]
        if not bos:
            cmd.append("--no-bos")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("llama-tokenize failed:\n" + r.stderr[-800:])
        for line in reversed(r.stdout.splitlines()):
            line = line.strip()
            if line.startswith("["):
                return json.loads(line)
        raise RuntimeError("cannot parse llama-tokenize output:\n" + r.stdout[-500:] + r.stderr[-500:])

    def encode(self, text, bos=False):
        key = (text, bos)
        if key not in self.cache:
            self.cache[key] = self._run(text, bos)
        return self.cache[key]

    def prefetch(self, items):
        todo = [k for k in set(items) if k not in self.cache]
        if not todo:
            return
        with ThreadPoolExecutor(8) as ex:
            for k, v in zip(todo, ex.map(lambda k: self._run(*k), todo)):
                self.cache[k] = v


# --------------------------------------------------------------------------- line segmentation
FENCE = re.compile(r"^\s*(```|~~~)")
HEAD = re.compile(r"^\s{0,3}#{1,6}[ \t]+")
LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])[ \t]+")
QUOTE = re.compile(r"^\s*>+[ \t]?")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def segment_lines(lines):
    """-> list of (text, is_markup, block_type).  Markup = syntax with no content of its own:
    heading '#' marks, list bullets, quote marks, code fences, rules, table separators, blank lines."""
    segs, in_fence = [], False
    for ln in lines:
        if FENCE.match(ln):
            segs.append((ln + "\n", 1, "code"))
            in_fence = not in_fence
            continue
        if in_fence:
            segs.append((ln + "\n", 0, "code"))
            continue
        if not ln.strip():
            segs.append((ln + "\n", 1, "blank"))
            continue
        if HR.match(ln):
            segs.append((ln + "\n", 1, "rule"))
            continue
        if "|" in ln and TABLE_SEP.match(ln):
            segs.append((ln + "\n", 1, "table"))
            continue
        for rx, bt in ((HEAD, "heading"), (LIST, "list"), (QUOTE, "quote")):
            m = rx.match(ln)
            if m:
                segs.append((m.group(0), 1, bt))
                segs.append((ln[m.end():] + "\n", 0, bt))
                break
        else:
            bt = "table" if ln.lstrip().startswith("|") else "paragraph"
            segs.append((ln + "\n", 0, bt))
    return segs


# --------------------------------------------------------------------------- tree
class Node:
    def __init__(self, title, level, lines, parent=None):
        self.title, self.level, self.lines, self.parent = title, level, lines, parent
        self.children = []
        self.own = []          # [(token, is_markup, block_type)]
        self.size = 0          # tokens in whole subtree (before merging)
        self.folded = False
        self.host = None
        self.eff = []
        self.folded_titles = []
        self.sid = None


def split_sections(md_text):
    lines = md_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    toks = MarkdownIt("commonmark").enable("table").parse(md_text)
    heads = []
    for i, t in enumerate(toks):
        if t.type == "heading_open":
            heads.append(dict(level=int(t.tag[1:]), start=t.map[0], end=t.map[1],
                              title=toks[i + 1].content.strip()))
    return lines, heads


def build_tree(md_text):
    lines, heads = split_sections(md_text)
    root = Node("ROOT", 0, lines[: heads[0]["start"]] if heads else lines)
    stack = [root]
    for k, h in enumerate(heads):
        nxt = heads[k + 1]["start"] if k + 1 < len(heads) else len(lines)
        while stack[-1].level >= h["level"]:
            stack.pop()
        n = Node(h["title"], h["level"], lines[h["start"]:nxt], stack[-1])
        stack[-1].children.append(n)
        stack.append(n)
    return root


def dfs(n):
    yield n
    for c in n.children:
        yield from dfs(c)


def tokenize_tree(root, tk, doc_title):
    segs = {}
    for n in dfs(root):
        s = segment_lines(n.lines)
        if n is root:
            s = [("Document: %s\n\n" % doc_title, 0, "document")] + s
        segs[id(n)] = s
    items = []
    for n in dfs(root):
        for j, (text, _, _) in enumerate(segs[id(n)]):
            items.append((text, n is root and j == 0))
    tk.prefetch(items)
    for n in dfs(root):
        for j, (text, mk, bt) in enumerate(segs[id(n)]):
            for t in tk.encode(text, bos=(n is root and j == 0)):
                n.own.append((t, mk, bt))


def compute_sizes(n):
    n.size = len(n.own) + sum(compute_sizes(c) for c in n.children)
    return n.size


def apply_fold(root, min_tokens):
    """Mark subtrees smaller than min_tokens as merged into the nearest kept ancestor. Returns #kept non-root nodes."""
    count = 0

    def rec(n, host):
        nonlocal count
        if n.parent is not None and n.size < min_tokens:
            n.folded, n.host = True, host
            for c in n.children:
                rec(c, host)
        else:
            n.folded, n.host = False, None
            if n.parent is not None:
                count += 1
            for c in n.children:
                rec(c, n)

    rec(root, None)
    return count


def flat(n):
    out = list(n.own)
    for c in n.children:
        out += flat(c)
    return out


def all_titles(n):
    return [x.title for x in dfs(n)]


def build_eff(n):
    n.eff = list(n.own)
    n.folded_titles = []
    for c in n.children:
        if c.folded:
            n.eff += flat(c)
            n.folded_titles += all_titles(c)
        else:
            build_eff(c)


def assign_sids(root):
    root.sid = 0
    k = 0
    for n in dfs(root):
        if n is not root and not n.folded:
            k += 1
            n.sid = k


def kept_descendant_sids(n):
    out = []
    for c in n.children:
        if not c.folded:
            out.append(c.sid)
            out += kept_descendant_sids(c)
    return out


# --------------------------------------------------------------------------- build document
def parse_markdown(md_text, tk, doc_title, min_tokens=24):
    root = build_tree(md_text)
    tokenize_tree(root, tk, doc_title)
    compute_sizes(root)
    mt = min_tokens
    while True:
        kept = apply_fold(root, mt)
        if kept + 1 <= MAX_SEQ:
            break
        mt = int(mt * 1.5) + 1
    build_eff(root)
    assign_sids(root)

    doc = dict(tokens=[], positions=[], seq_ids=[], node_type=[], parent=[], is_markup=[], nodes=[])

    def emit(n, start, depth):
        ids = [n.sid] + kept_descendant_sids(n)
        psid = n.parent_sid if hasattr(n, "parent_sid") else -1
        for i, (t, mk, bt) in enumerate(n.eff):
            doc["tokens"].append(t)
            doc["positions"].append(start + i)
            doc["seq_ids"].append(list(ids))
            doc["node_type"].append(bt)
            doc["parent"].append(psid)
            doc["is_markup"].append(mk)
        meta = dict(sid=n.sid, title=n.title if n.parent else "(document)", level=n.level,
                    parent_sid=psid, depth=depth, start=start, own_tokens=len(n.eff),
                    contains=n.folded_titles)
        doc["nodes"].append(meta)
        sub = len(n.eff)
        for c in n.children:
            if not c.folded:
                c.parent_sid = n.sid
                sub += emit(c, start + len(n.eff), depth + 1)
        meta["subtree_tokens"] = sub
        return sub

    emit(root, 0, 0)
    heading_depth = 0
    for n in dfs(root):
        d, p = 0, n
        while p.parent is not None:
            d, p = d + 1, p.parent
        heading_depth = max(heading_depth, d)
    doc["meta"] = dict(doc_title=doc_title, min_tokens_used=mt, max_heading_depth=heading_depth,
                       n_sections_before_merge=sum(1 for _ in dfs(root)) - 1)
    validate(doc)
    return doc


def validate(doc):
    per = {}
    for pos, ids in zip(doc["positions"], doc["seq_ids"]):
        for s in ids:
            per.setdefault(s, []).append(pos)
    for s, ps in per.items():
        if sorted(ps) != list(range(len(ps))):
            raise ValueError("positions of seq id %d are not 0..n-1 without gaps/duplicates" % s)
    if max(per) >= MAX_SEQ:
        raise ValueError("seq id %d >= n_seq_max %d" % (max(per), MAX_SEQ))
    if len(doc["tokens"]) > 1900:
        print("warning: %d tokens (keep documents under 2000)" % len(doc["tokens"]), file=sys.stderr)


# --------------------------------------------------------------------------- test helper: change one section
def perturb(md_text, title):
    """Replace the body of section `title` by different, longer text (headings untouched)."""
    lines, heads = split_sections(md_text)
    for k, h in enumerate(heads):
        if h["title"] == title:
            nxt = heads[k + 1]["start"] if k + 1 < len(heads) else len(lines)
            body = lines[h["end"]:nxt]
            new = [codecs.encode(l, "rot13") for l in body]
            new += ["", "Unrelated filler: the weather stayed calm and the garden was green all spring.", ""]
            return "\n".join(lines[: h["end"]] + new + lines[nxt:]) + "\n"
    raise KeyError(title)


def doc_title_of(md_text, path):
    _, heads = split_sections(md_text)
    for h in heads:
        if h["level"] == 1:
            return h["title"]
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--tokenizer", default="build/bin/llama-tokenize")
    ap.add_argument("--fake-tokenizer", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=24)
    ap.add_argument("--perturb", help="title of a section whose body should be replaced (for Test 1)")
    a = ap.parse_args()
    if not a.fake_tokenizer and not a.model:
        ap.error("--model is required unless --fake-tokenizer is used")
    text = open(a.md, encoding="utf-8").read()
    if a.perturb:
        text = perturb(text, a.perturb)
    tk = Tokenizer(a.tokenizer, a.model, a.fake_tokenizer)
    doc = parse_markdown(text, tk, doc_title_of(text, a.md), a.min_tokens)
    with open(a.out, "w") as f:
        json.dump(doc, f)
    print("%s: %d tokens, %d sequences, min_tokens=%d" % (a.out, len(doc["tokens"]), len(doc["nodes"]),
                                                          doc["meta"]["min_tokens_used"]))


if __name__ == "__main__":
    main()
