#!/usr/bin/env python3
"""
Syntax-only stand-in for the Roslyn extractor, built on tree-sitter. Emits the same JSONL
schema so dupdetect.py can run without .NET. Use the Roslyn extractor for real audits:
without a semantic model this version identifies locals/fields by name only.

    pip install tree-sitter tree-sitter-c-sharp
    python ts_extract.py <source-root> -o chunks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import tree_sitter_c_sharp as tscs
from tree_sitter import Language, Node, Parser

LANG = Language(tscs.language())
SKIP_DIRS = {"bin", "obj", ".git", ".vs", ".idea", "node_modules", "TestResults"}
GEN_SUFFIXES = (".g.cs", ".g.i.cs", ".designer.cs", ".generated.cs", ".assemblyinfo.cs", ".assemblyattributes.cs")
TEST_ATTRS = {"Fact", "Theory", "Test", "TestMethod", "DataTestMethod", "TestCase", "TestCaseSource", "SetUp",
              "TearDown", "OneTimeSetUp", "OneTimeTearDown", "TestInitialize", "TestCleanup", "TestFixture", "TestClass"}
TEST_NAME = re.compile(r"(?i)(tests?|specs?|\.testing)$")
GUARD_PREFIXES = ("ArgumentNullException.ThrowIf", "ArgumentException.ThrowIf", "ArgumentOutOfRangeException.ThrowIf",
                  "ObjectDisposedException.ThrowIf", "Guard.", "Ensure.", "Requires.", "Debug.Assert", "Contract.Requires")
PC_CALLS = {"OnPropertyChanged", "RaisePropertyChanged", "NotifyPropertyChanged", "OnPropertyChanging", "SetProperty",
            "SetField", "Set", "RaiseAndSetIfChanged", "NotifyOfPropertyChange"}
DI_TYPES = {"IServiceCollection", "IApplicationBuilder", "WebApplicationBuilder", "WebApplication", "IHostBuilder",
            "IHostApplicationBuilder", "IEndpointRouteBuilder", "ContainerBuilder", "IMvcBuilder"}
MAPPING_NAME = re.compile(r"^(Map|To[A-Z]|From[A-Z]|Convert|Project[A-Z])")
REPO_TYPE = re.compile(r"(Repository|Repo|Store|Dao|DataAccess)$")
TYPE_DECLS = {"class_declaration", "struct_declaration", "record_declaration", "interface_declaration",
              "record_struct_declaration"}
METHOD_LIKE = {"method_declaration", "constructor_declaration", "destructor_declaration", "operator_declaration",
               "conversion_operator_declaration", "local_function_statement"}
DECL_PARENTS = {("variable_declarator", "name"), ("parameter", "name"), ("foreach_statement", "left"),
                ("catch_declaration", "name"), ("from_clause", "name"), ("let_clause", "name"),
                ("join_clause", "name"), ("join_into_clause", "name"), ("query_continuation", "name")}


def txt(n: Node | None) -> str:
    return n.text.decode("utf-8", "replace") if n is not None else ""


def field_of(n: Node) -> str | None:
    p = n.parent
    if p is None:
        return None
    for i, c in enumerate(p.children):
        if c.id == n.id:
            return p.field_name_for_child(i)
    return None


def walk(n: Node):
    yield n
    for c in n.children:
        yield from walk(c)


# --------------------------------------------------------------------------------------

class Site:
    def __init__(self, node, kind, name, sig, prefix, block, arrow, is_void):
        self.node, self.kind, self.name, self.sig = node, kind, name, sig
        self.prefix, self.block, self.arrow, self.is_void = prefix, block, arrow, is_void


def returns_void(ret: Node | None, is_async: bool) -> bool:
    if ret is None:
        return True
    t = txt(ret)
    return t == "void" or (is_async and t in ("Task", "ValueTask", "System.Threading.Tasks.Task"))


def modifiers(n: Node) -> set[str]:
    return {txt(c) for c in n.children if c.type == "modifier"}


def describe(n: Node) -> Site | None:
    t = n.type
    if t in METHOD_LIKE:
        body = n.child_by_field_name("body")
        if body is None:
            return None
        block = body if body.type == "block" else None
        arrow = body if body.type == "arrow_expression_clause" else None
        params = n.child_by_field_name("parameters")
        prefix = [params] if params is not None else []
        mods = modifiers(n)
        if t == "method_declaration" or t == "local_function_statement":
            ret = n.child_by_field_name("returns") or n.child_by_field_name("type")
            name = txt(n.child_by_field_name("name"))
            tp = next((c for c in n.children if c.type == "type_parameter_list"), None)
            name += txt(tp)
            kind = "method" if t == "method_declaration" else "localFunction"
            return Site(n, kind, name, f"{txt(ret)} {name}{txt(params)}", prefix, block, arrow,
                        returns_void(ret, "async" in mods))
        if t == "constructor_declaration":
            init = next((c for c in n.children if c.type == "constructor_initializer"), None)
            if init is not None:
                prefix.append(init)
            return Site(n, "constructor", ".cctor" if "static" in mods else ".ctor",
                        f"{txt(n.child_by_field_name('name'))}{txt(params)}", prefix, block, arrow, True)
        if t == "destructor_declaration":
            return Site(n, "destructor", "Finalize", "~()", prefix, block, arrow, True)
        op = n.child_by_field_name("operator")
        name = f"operator {txt(op)}" if op is not None else "operator"
        return Site(n, "operator", name, f"{name}{txt(params)}", prefix, block, arrow, False)

    if t == "accessor_declaration":
        body = n.child_by_field_name("body")
        if body is None:
            return None
        owner = n.parent.parent if n.parent is not None else None
        owner_name = "this[]" if owner is not None and owner.type == "indexer_declaration" else txt(owner.child_by_field_name("name")) if owner is not None else "?"
        kw = txt(n.child_by_field_name("name"))
        prefix = []
        if owner is not None and owner.type == "indexer_declaration" and owner.child_by_field_name("parameters") is not None:
            prefix.append(owner.child_by_field_name("parameters"))
        return Site(n, "accessor", f"{owner_name}.{kw}", f"{owner_name}.{kw}", prefix,
                    body if body.type == "block" else None, body if body.type == "arrow_expression_clause" else None,
                    kw != "get")

    if t in ("property_declaration", "indexer_declaration"):
        arrow = n.child_by_field_name("value") if t == "property_declaration" else None
        if arrow is None:
            arrow = next((c for c in n.children if c.type == "arrow_expression_clause"), None)
        if arrow is None or arrow.type != "arrow_expression_clause":
            return None
        name = txt(n.child_by_field_name("name")) if t == "property_declaration" else "this[]"
        prefix = [n.child_by_field_name("parameters")] if t == "indexer_declaration" and n.child_by_field_name("parameters") else []
        return Site(n, "accessor", f"{name}.get", name, prefix, None, arrow, False)

    if t in ("lambda_expression", "anonymous_method_expression"):
        body = n.child_by_field_name("body") or next((c for c in n.children if c.type == "block"), None)
        if body is None or body.type != "block":
            return None
        params = n.child_by_field_name("parameters")
        return Site(n, "lambda", "<lambda>", "<lambda>", [params] if params is not None else [], body, None, False)
    return None


# --------------------------------------------------------------------------------------

def is_guard(st: Node) -> bool:
    if st.type == "expression_statement" and st.named_child_count and st.named_children[0].type == "invocation_expression":
        callee = re.sub(r"\s+", "", txt(st.named_children[0].child_by_field_name("function")))
        callee = callee.removeprefix("System.")
        return callee.startswith(GUARD_PREFIXES)
    if st.type == "if_statement" and st.child_by_field_name("alternative") is None:
        c = st.child_by_field_name("consequence")
        if c is None:
            return False
        if c.type == "throw_statement":
            return True
        stmts = [x for x in c.named_children if x.type != "comment"] if c.type == "block" else []
        return len(stmts) == 1 and stmts[0].type == "throw_statement"
    return False


def statements(block: Node | None) -> list[Node]:
    return [c for c in block.named_children if c.type != "comment"] if block is not None else []


class Normalizer:
    def __init__(self, field_names: set[str], type_params: set[str] = frozenset()):
        self.fields = field_names
        self.tparams = type_params
        self.locals: set[str] = set()
        self.map: dict[str, str] = {}
        self.counters = {"v": 0, "f": 0, "g": 0}
        self.out: list[str] = []
        self.skip_ids: set[int] = set()
        self.guards: list[Node] = []

    def ph(self, name: str, prefix: str) -> str:
        key = prefix + ":" + name
        if key not in self.map:
            self.counters[prefix] += 1
            self.map[key] = f"{prefix}{self.counters[prefix]}"
        return self.map[key]

    def run(self, site: Site) -> list[str]:
        regions = list(site.prefix) + [r for r in (site.block, site.arrow) if r is not None]
        for r in regions:
            for n in walk(r):
                if n.type == "identifier" and (n.parent.type, field_of(n)) in DECL_PARENTS:
                    self.locals.add(txt(n))
                elif n.type == "implicit_parameter":
                    self.locals.add(txt(n))
                elif n.type == "identifier" and n.parent.type in ("declaration_pattern", "var_pattern", "parenthesized_variable_designation", "declaration_expression") and field_of(n) in ("name", None):
                    self.locals.add(txt(n))
        if site.block is not None:
            for st in statements(site.block):
                if not is_guard(st):
                    break
                self.guards.append(st)
                self.skip_ids.add(st.id)
        for p in site.prefix:
            self.emit(p)
        if site.block is not None:
            self.emit(site.block)
        elif site.arrow is not None:
            self.out.append("{")
            if not site.is_void:
                self.out.append("return")
            for c in site.arrow.children:
                if c.type != "=>":
                    self.emit(c)
            self.out += [";", "}"]
        return self.out

    def emit(self, n: Node) -> None:
        t = n.type
        if n.id in self.skip_ids or t == "comment":
            return
        if t == "member_access_expression":
            expr = n.child_by_field_name("expression")
            if expr is not None and expr.type == "this":
                self.emit_name(n.child_by_field_name("name"), member_of_this=True)
                return
        if t == "invocation_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "member_access_expression" and txt(fn.child_by_field_name("name")) == "ConfigureAwait":
                self.emit(fn.child_by_field_name("expression"))
                return
        if t == "variable_declaration" and n.parent is not None and n.parent.type in ("local_declaration_statement", "for_statement", "using_statement", "fixed_statement"):
            for c in n.children:
                if field_of(c) == "type":
                    self.out.append("T")
                else:
                    self.emit(c)
            return
        if t in ("foreach_statement", "declaration_expression"):
            for c in n.children:
                if field_of(c) == "type":
                    self.out.append("T")
                else:
                    self.emit(c)
            return
        if t in ("string_literal", "verbatim_string_literal", "raw_string_literal"):
            self.out.append("STR")
            return
        if t == "character_literal":
            self.out.append("CHR")
            return
        if t in ("integer_literal", "real_literal"):
            self.out.append("NUM")
            return
        if t in ("string_content", "interpolated_string_text", "string_literal_content"):
            self.out.append("STR")
            return
        if n.child_count == 0:
            if t in ("identifier", "implicit_parameter"):
                self.emit_name(n)
            else:
                self.out.append(txt(n))
            return
        for c in n.children:
            self.emit(c)

    def emit_name(self, n: Node | None, member_of_this: bool = False) -> None:
        if n is None:
            return
        name, parent, fld = txt(n), n.parent, field_of(n)
        if not member_of_this:
            if parent.type == "member_access_expression" and fld == "name":
                self.out.append(name); return
            if parent.type in ("member_binding_expression", "qualified_name", "alias_qualified_name"):
                self.out.append(name); return
            if parent.type == "argument" and fld == "name":
                self.out.append(name); return
            if name in self.locals:
                self.out.append(self.ph(name, "v")); return
        if name in self.fields:
            self.out.append(self.ph(name, "f")); return
        if name in self.tparams and not member_of_this:
            self.out.append(self.ph(name, "g")); return
        self.out.append(name)

    def count_statements(self, site: Site) -> int:
        if site.block is None:
            return 1 if site.arrow is not None else 0
        guard_ranges = [(g.start_byte, g.end_byte) for g in self.guards]
        n = 0
        for x in walk(site.block):
            if x.type.endswith("_statement") and x.type != "block" and x.is_named:
                if not any(a <= x.start_byte and x.end_byte <= b for a, b in guard_ranges):
                    n += 1
        return n


# --------------------------------------------------------------------------------------

def raw_tokens(site: Site) -> list[str]:
    toks = []
    for r in list(site.prefix) + [x for x in (site.block, site.arrow) if x is not None]:
        toks += [txt(x) for x in walk(r) if x.child_count == 0 and x.type != "comment"]
    return toks


def attr_names(n: Node) -> set[str]:
    out = set()
    cur = n
    while cur is not None:
        for c in cur.children:
            if c.type == "attribute_list":
                for a in c.named_children:
                    if a.type == "attribute":
                        nm = txt(a.child_by_field_name("name")).split(".")[-1]
                        out.add(nm[:-9] if nm.endswith("Attribute") and len(nm) > 9 else nm)
        cur = cur.parent
    return out


def enclosing(n: Node, types: set[str]):
    cur = n.parent
    while cur is not None:
        if cur.type in types:
            yield cur
        cur = cur.parent


def field_names(n: Node) -> set[str]:
    t = next(enclosing(n, TYPE_DECLS), None)
    out = set()
    if t is None:
        return out
    body = t.child_by_field_name("body")
    for m in (body.named_children if body is not None else []):
        if m.type == "field_declaration" and "const" not in modifiers(m):
            for x in walk(m):
                if x.type == "identifier" and x.parent.type == "variable_declarator" and field_of(x) == "name":
                    out.add(txt(x))
    return out


def type_param_names(n: Node) -> set[str]:
    out = set()
    cur = n
    while cur is not None:
        if cur.type in METHOD_LIKE or cur.type in TYPE_DECLS:
            for c in cur.children:
                if c.type == "type_parameter_list":
                    out |= {txt(tp.child_by_field_name("name")) for tp in c.named_children if tp.type == "type_parameter"}
        cur = cur.parent
    return out


def type_chain(n: Node) -> str:
    parts = [txt(t.child_by_field_name("name")) for t in enclosing(n, TYPE_DECLS)]
    return ".".join(reversed(parts)) or "(global)"


def namespace_of(n: Node) -> str:
    ns = [txt(x.child_by_field_name("name")) for x in enclosing(n, {"namespace_declaration"})]
    if not ns:
        root = n
        while root.parent is not None:
            root = root.parent
        fs = next((c for c in root.children if c.type == "file_scoped_namespace_declaration"), None)
        if fs is not None:
            ns = [txt(fs.child_by_field_name("name"))]
    return ".".join(reversed(ns))


def is_forwarder(site: Site, core: list[Node]) -> bool:
    if site.kind == "constructor":
        return site.block is not None and not core and any(p.type == "constructor_initializer" for p in site.prefix)
    e = None
    if site.arrow is not None:
        e = next((c for c in site.arrow.named_children), None)
    elif len(core) == 1 and core[0].type in ("return_statement", "expression_statement"):
        e = next((c for c in core[0].named_children), None)
    if e is not None and e.type == "await_expression":
        e = e.named_children[-1] if e.named_child_count else None
    if e is not None and e.type == "invocation_expression":
        fn = e.child_by_field_name("function")
        if fn is not None and fn.type == "member_access_expression" and txt(fn.child_by_field_name("name")) == "ConfigureAwait":
            e = fn.child_by_field_name("expression")
    if e is None or e.type != "invocation_expression":
        return False
    args = e.child_by_field_name("arguments")
    for a in (args.named_children if args is not None else []):
        v = a.named_children[-1] if a.named_child_count else None
        if v is None:
            continue
        if v.type in ("identifier", "this", "boolean_literal", "integer_literal", "null_literal", "string_literal", "default_expression"):
            continue
        if v.type == "member_access_expression" and v.child_by_field_name("expression").type in ("identifier", "this"):
            continue
        return False
    return True


def is_trivial_ctor(site: Site, core: list[Node]) -> bool:
    if site.kind != "constructor" or not core:
        return False
    for st in core:
        if st.type != "expression_statement" or not st.named_child_count or st.named_children[0].type != "assignment_expression":
            return False
        a = st.named_children[0]
        if txt(a.child_by_field_name("operator")) != "=":
            return False
        r = a.child_by_field_name("right")
        if r.type == "binary_expression" and txt(r.child_by_field_name("operator")) == "??" and r.child_by_field_name("right").type == "throw_expression":
            r = r.child_by_field_name("left")
        if r.type not in ("identifier", "integer_literal", "string_literal", "boolean_literal", "null_literal"):
            if not (r.type in ("object_creation_expression", "implicit_object_creation_expression") and re.fullmatch(r"new\s*\w*(<.*>)?\s*\(\s*\)", txt(r))):
                return False
    return True


def classify(site: Site, core, attrs, test_path, startup, rel) -> tuple[list[str], list[str]]:
    flags, hints = [], []
    base = site.name.split("<")[0]
    if test_path or attrs & TEST_ATTRS or any(p.endswith("Tests") for p in type_chain(site.node).split(".")):
        flags.append("test")
    if site.block is not None and not core:
        flags.append("empty")
    if is_forwarder(site, core):
        flags.append("forwarder")
    if is_trivial_ctor(site, core):
        flags.append("trivialCtor")
    if base in ("Equals", "GetHashCode", "CompareTo") or (site.kind == "operator" and site.name.split(" ", 1)[-1] in ("==", "!=", "<", ">", "<=", ">=")):
        flags.append("equality")
    if base == "ToString" and site.kind == "method":
        flags.append("toString")
    if base in ("Dispose", "DisposeAsync", "DisposeCore", "DisposeAsyncCore", "Finalize"):
        flags.append("dispose")
    body_text = txt(site.node)
    if (site.kind == "method" and base in PC_CALLS and "PropertyChang" in body_text) or \
       (site.kind == "accessor" and site.name.endswith((".set", ".init")) and
        (any(re.search(rf"\b{c}\s*(<[^>]*>)?\s*\(", body_text) for c in PC_CALLS) or "PropertyChanged?.Invoke" in body_text)):
        flags.append("propertyChanged")
    if site.kind == "method":
        params = site.node.child_by_field_name("parameters")
        first = next((c for c in params.named_children if c.type == "parameter"), None) if params is not None else None
        ftype = txt(first.child_by_field_name("type")).split(".")[-1] if first is not None else ""
        if base in ("ConfigureServices", "ConfigureContainer", "RegisterServices", "AddServices") or ftype in DI_TYPES:
            flags.append("diWiring")
    if startup:
        flags.append("startup")
    if MAPPING_NAME.match(base) and re.search(r"new\s*\w*\s*(\(\s*\))?\s*\{(?:[^{}]*=){3}", body_text):
        hints.append("mapping")
    t = next(enclosing(site.node, TYPE_DECLS), None)
    if t is not None and REPO_TYPE.search(txt(t.child_by_field_name("name"))):
        hints.append("repository")
    if any(x.type == "query_expression" for x in walk(site.node)):
        hints.append("linqQuery")
    return flags, hints


def extract_code(src: bytes, n: Node) -> str:
    line_start = src.rfind(b"\n", 0, n.start_byte) + 1
    lead = src[line_start:n.start_byte].decode("utf-8", "replace")
    raw = "".join("\t" if ch == "\t" else " " for ch in lead) + src[n.start_byte:n.end_byte].decode("utf-8", "replace")
    lines = raw.replace("\r\n", "\n").split("\n")
    widths = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    ind = min(widths) if widths else 0
    return "\n".join(l[ind:] if not l[:ind].strip() else l.lstrip() for l in lines)


def code_lines(code: str) -> int:
    return sum(1 for l in code.split("\n")
               if (s := l.strip()) and not s.startswith(("//", "/*", "*")) and s not in ("{", "}"))


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def is_generated(path: Path, src: bytes) -> bool:
    name = path.name.lower()
    if name.endswith(GEN_SUFFIXES) or name == "assemblyinfo.cs":
        return True
    if "/migrations/" in path.as_posix().lower():
        return True
    head = src[:2000].decode("utf-8", "replace").lower()
    return "<auto-generated" in head or "<autogenerated" in head


def find_project(path: Path, root: Path, cache: dict) -> str:
    d = path.parent
    chain = []
    while True:
        if d in cache:
            name = cache[d]
            break
        chain.append(d)
        projs = sorted(d.glob("*.csproj"))
        if projs:
            name = projs[0].stem
            break
        if d == root or d.parent == d:
            name = "(no project)"
            break
        d = d.parent
    for c in chain:
        cache[c] = name
    return name


def extract_file(path: Path, root: Path, project: str, parser: Parser, min_tokens: int):
    src = path.read_bytes()
    if src.startswith(b"\xef\xbb\xbf"):
        src = src[3:]
    tree = parser.parse(src)
    rel = path.relative_to(root).as_posix()
    test_path = bool(TEST_NAME.search(project)) or any(TEST_NAME.search(s) and len(s) <= 30 for s in rel.split("/")[:-1])
    startup = path.name in ("Program.cs", "Startup.cs")
    for n in walk(tree.root_node):
        site = describe(n)
        if site is None:
            continue
        attrs = attr_names(n)
        if attrs & {"GeneratedCode", "CompilerGenerated"}:
            continue
        norm = Normalizer(field_names(n), type_param_names(n))
        toks = norm.run(site)
        if len(toks) < min_tokens:
            continue
        parent = next((s for s in (describe(a) for a in enclosing(n, METHOD_LIKE | {"accessor_declaration", "property_declaration", "indexer_declaration", "lambda_expression", "anonymous_method_expression"})) if s), None)
        start, end = n.start_point[0] + 1, n.end_point[0] + 1
        member = site.name
        if site.kind == "lambda":
            member = f"{parent.name if parent else '?'}.<lambda@L{start}>"
        elif site.kind == "localFunction":
            member = f"{parent.name if parent else '?'}.{site.name}"
        core = statements(site.block)[len(norm.guards):] if site.block is not None else []
        flags, hints = classify(site, core, attrs, test_path, startup, rel)
        code = extract_code(src, n)
        normalized = " ".join(toks)
        yield {
            "id": f"{rel}:{start}:{n.start_point[1] + 1}:{site.kind}",
            "file": rel, "project": project, "namespace": namespace_of(n), "type": type_chain(n),
            "member": member, "kind": site.kind, "signature": re.sub(r"\s+", " ", site.sig).strip(),
            "startLine": start, "endLine": end, "codeLines": code_lines(code), "tokens": len(toks),
            "statements": norm.count_statements(site),
            "parentId": f"{rel}:{parent.node.start_point[0] + 1}:{parent.node.start_point[1] + 1}:{parent.kind}" if parent else None,
            "flags": flags, "hints": hints,
            "exactHash": h(" ".join(raw_tokens(site))), "normalizedHash": h(normalized),
            "code": code, "normalized": normalized,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--min-tokens", type=int, default=12)
    ap.add_argument("--exclude-dir", action="append", default=[])
    args = ap.parse_args()
    root = args.root.resolve()
    skip = SKIP_DIRS | set(args.exclude_dir)
    parser = Parser(LANG)
    cache: dict = {}
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        files += [Path(dirpath) / f for f in sorted(filenames) if f.endswith(".cs")]
    out, generated = [], 0
    for f in files:
        src_head = f.read_bytes()[:2000]
        if is_generated(f.relative_to(root), src_head):
            generated += 1
            continue
        out += list(extract_file(f, root, find_project(f, root, cache), parser, args.min_tokens))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as w:
        for c in out:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(files)} files ({generated} generated skipped) -> {len(out)} chunks in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
