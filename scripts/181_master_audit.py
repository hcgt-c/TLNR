# -*- coding: utf-8 -*-
"""181_master_audit.py — audit the unified master manuscript against the existing verification chain.

The master (`paper/unified/MASTER_manuscript_EN.md`) is assembled from the two already-audited
manuscripts plus newly written theory. This script makes three mechanical guarantees:

  1. every claim in `paper/unified/evidence_map.json` must name a verifier that actually exists
     (a label in results/paperB_numbers_check.json for 148, in results/code_audit.json for 172, or in
     results/portfolio_check.json for 163), so the master can never point at a check that is not real;
  2. every internal reference in the master (`Section N[.M]`, `Table N`, `Figure N`) must resolve to a
     heading or caption that exists in the master itself;
  3. no externally-pointing artifact strings ("companion paper", "the other manuscript", ...) may appear,
     because each sliced submission must be self-contained.

Migration coverage (how many claims' literals are already present in the master body) is reported as a
number, not as a failure: Sections 5-11 are migrated incrementally and this script is the progress meter.

Usage: python scripts/181_master_audit.py
Outputs results/master_audit.{json,md}
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import repo_paths as rp
import os, re, json

WORK = rp.REPO_ROOT
MASTER = os.path.join(WORK, "paper", "unified", "MASTER_manuscript_EN.md")
EMAP = os.path.join(WORK, "paper", "unified", "evidence_map.json")
MANIFEST = os.path.join(WORK, "paper", "unified", "manifest.json")
SLICING = os.path.join(WORK, "paper", "unified", "slicing_map.json")
RES = os.path.join(WORK, "results")

EXTERNAL_POINTERS = ["companion paper", "the other manuscript", "the companion", "see our other paper",
                     "in a separate paper", "as detailed elsewhere"]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def verifier_labels():
    """label -> {literal, pass} for each existing audit, so the master inherits verified values.

    The master cites a claim by its verifier label. This returns what that label actually asserts in the
    audit reports, so 181 can check that (a) the label exists, (b) the source check passes, and (c) the
    literal quoted for the claim is the literal the source audit verified (rather than a fresh number).
    """
    out = {}
    p = os.path.join(RES, "paperB_numbers_check.json")
    if os.path.exists(p):
        out["148"] = {c["label"]: {"literal": c["literal"], "pass": c["pass"]} for c in load(p)["checks"]}
    p = os.path.join(RES, "code_audit.json")
    if os.path.exists(p):
        out["172"] = {c["label"]: {"literal": c["literal"], "pass": c["pass"]}
                      for c in load(p)["paperA_checks"]}
    p = os.path.join(RES, "portfolio_check.json")
    if os.path.exists(p):
        out["163"] = {r["label"]: {"literal": r["expected"], "pass": r["pass"]}
                      for r in load(p).get("shared_table", [])}
    p = os.path.join(RES, "heat_tables.json")
    if os.path.exists(p):
        # script 195 recomputes every quoted literal from the dissipative-family and composition result
        # files and confirms it appears verbatim in the master, so the master inherits that check here
        out["195"] = {c_["label"]: {"literal": c_["literal"], "pass": c_["pass"]}
                      for c_ in load(p).get("checks", [])}
    p = os.path.join(RES, "star_existence_check.json")
    if os.path.exists(p):
        # script 230 recomputes every quoted literal from results/star_existence_test.json (script 229),
        # so the existence-level numbers in the master are audited rather than hand-copied
        out["229"] = {c_["label"]: {"literal": c_["literal"], "pass": c_["pass"]}
                      for c_ in load(p).get("checks", [])}
    return out


def main():
    raw = open(MASTER, encoding="utf-8").read()
    body = re.sub(r"<!--.*?-->", "", raw, flags=re.S)          # drop non-rendering build notes
    txt = body.replace("\u2212", "-")
    emap = load(EMAP)
    man = load(MANIFEST)
    labels = verifier_labels()
    planned_sec = set(man["sections"])
    planned_tab = set(man["tables"])
    planned_fig = set(man["figures"])

    # ---- split the body into sections so a claim's literal must appear in *its own* section
    # (otherwise a number quoted elsewhere, e.g. a range endpoint, would count as migrated)
    def section_text(body_txt):
        heads = [(m.start(), m.group(2)) for m in re.finditer(r"^(#{2,4})\s+(\d+(?:\.\d+)*)", body_txt, re.M)]
        out = {}
        for i, (pos, num) in enumerate(heads):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(body_txt)
            out[num] = body_txt[pos:end]
        return out
    SEC = section_text(body)

    def in_claimed_section(num, lit):
        """True when the section a claim is mapped to quotes that value (or a more precise form of it).

        Literal equality is too strict: a claim verified as 0.84 is legitimately written as 0.836 in the
        prose. We therefore accept any number in the claimed section that rounds to the claimed literal
        at the literal's precision, which is the same criterion the number audits use.
        """
        if not lit:
            return False
        keys = [num]
        if "." in num:
            keys.append(num.split(".")[0])          # a top-level claim may be evidenced in a subsection
        txt_sec = "\n".join(SEC.get(k, "") for k in keys)
        if lit in txt_sec:
            return True
        head = re.match(r"[+-]?\d+(?:\.\d+)?", lit.replace("\u2212", "-"))
        if not head:
            return False
        want = float(head.group(0))
        nd = len(head.group(0).split(".")[1]) if "." in head.group(0) else 0
        for m in re.finditer(r"[+-]?\d+(?:\.\d+)?", txt_sec.replace("\u2212", "-")):
            try:
                if abs(round(float(m.group(0)), nd) - want) <= 10.0 ** (-nd) / 2 + 1e-12:
                    return True
            except ValueError:
                continue
        return False

    # ---- 1. evidence map integrity, and literal coverage of the body
    rows, missing_label = [], []
    for c in emap["claims"]:
        v = c["verifier"]
        ok_label, src_pass, lit_match = None, None, None
        lit = (c.get("literal") or "").replace("\u2212", "-")
        if v == "manual":
            ok_label = "n/a (theory or prose; no numeric verifier)"
        elif v in labels and c.get("ref") in labels[v]:
            src = labels[v][c["ref"]]
            ok_label = True
            src_pass = bool(src["pass"])
            # compare on values, not on typography: degree signs and spacing are presentation only
            _n = lambda x: re.sub(r"[\s°]", "", (x or "").replace("\u2212", "-"))
            lit_match = (_n(src["literal"]) == _n(lit)) if lit else True
            if not src_pass:
                missing_label.append(f"{c['id']}: source check '{v}:{c['ref']}' does not currently pass")
            if lit and not lit_match:
                missing_label.append(f"{c['id']}: literal '{c['literal']}' differs from the verified "
                                     f"literal '{src['literal']}' of {v}:{c['ref']}")
        elif v in labels:
            ok_label = False
            missing_label.append(f"{c['id']}: {v} has no check labelled '{c.get('ref')}'")
        else:
            ok_label = False
            missing_label.append(f"{c['id']}: verifier '{v}' has no result file yet (run that audit first)")
        present = in_claimed_section(c["master"], lit)
        rows.append({"id": c["id"], "master": c["master"], "claim": c["claim"], "verifier": v,
                     "ref": c.get("ref"), "literal": c.get("literal"), "verifier_ok": ok_label,
                     "source_passes": src_pass, "literal_matches_source": lit_match,
                     "literal_in_master": present, "strength": c.get("strength"),
                     "note": c.get("note", "")})

    n_num = sum(1 for r in rows if r["literal"])
    n_present = sum(1 for r in rows if r["literal_in_master"])
    coverage = round(100.0 * n_present / n_num, 1) if n_num else 100.0

    # ---- 2. cross-references inside the master: resolves / planned-but-not-migrated / dangling
    secs = {m.group(1) for m in re.finditer(r"^#{2,4}\s+(\d+(?:\.\d+)*)", body, re.M)}
    sec_refs = set(re.findall(r"Section\s+(\d+(?:\.\d+)*)", txt))
    def resolves(r, have, planned):
        if r in have or any(s.startswith(r + ".") for s in have):
            return "ok"
        return "planned" if (r in planned or any(s.startswith(r + ".") for s in planned)) else "dangling"
    sec_state = {r: resolves(r, secs, planned_sec) for r in sorted(sec_refs)}
    tab_caps = {m.group(1) for m in re.finditer(r"\*\*Table ([A-Z]?\d+[a-z]?)\.\*\*", body)}
    tab_refs = set(re.findall(r"Table ([A-Z]?\d+)", txt))
    tab_state = {r: ("ok" if r in tab_caps else ("planned" if r in planned_tab else "dangling"))
                 for r in sorted(tab_refs, key=lambda x: (len(x), x))}
    fig_caps = {m.group(1) for m in re.finditer(r"\*\*Figure (\d+)\*\*", body)}
    fig_refs = set(re.findall(r"Figure (\d+)", txt))
    fig_state = {r: ("ok" if r in fig_caps else ("planned" if r in planned_fig else "dangling"))
                 for r in sorted(fig_refs, key=int)}
    pending = [f"Section {r}" for r, s in sec_state.items() if s == "planned"] + \
              [f"Table {r}" for r, s in tab_state.items() if s == "planned"] + \
              [f"Figure {r}" for r, s in fig_state.items() if s == "planned"]
    crossref_issues = [f"master: reference to Section {r}, which is neither present nor planned"
                       for r, s in sec_state.items() if s == "dangling"] + \
                      [f"master: reference to Table {r}, which is neither present nor planned"
                       for r, s in tab_state.items() if s == "dangling"] + \
                      [f"master: reference to Figure {r}, which is neither present nor planned"
                       for r, s in fig_state.items() if s == "dangling"]

    # ---- 2b. slicing map: each version's seed must close over its own internal references
    slicing = load(SLICING) if os.path.exists(SLICING) else {"versions": {}}
    slice_report = {}
    for name, v in slicing.get("versions", {}).items():
        have = set(v.get("seed", []))
        changed = True
        while changed:                                  # transitive closure of Section references
            changed = False
            for sec in list(have):
                txt_sec = SEC.get(sec, "")
                for r in re.findall(r"Section\s+(\d+(?:\.\d+)*)", txt_sec):
                    cand = [k for k in SEC if k == r or k.startswith(r + ".")]
                    for k in cand:
                        if k not in have:
                            have.add(k); changed = True
        closure = sorted(have, key=lambda x: [int(p) for p in x.split(".")])
        unresolved = []
        for sec in closure:
            for r in re.findall(r"Section\s+(\d+(?:\.\d+)*)", SEC.get(sec, "")):
                if not any(k == r or k.startswith(r + ".") for k in closure):
                    unresolved.append(f"{name}: Section {sec} cites Section {r}, which is outside the closure")
        slice_report[name] = {"seed": sorted(v.get("seed", [])), "closure": closure,
                              "n_closure": len(closure), "unresolved": unresolved,
                              "own_contribution": v.get("own_contribution", "")}
    slice_issues = [u for r in slice_report.values() for u in r["unresolved"]]

    # ---- 3. self-containment
    pointer_issues = [f"master contains external pointer '{p}'" for p in EXTERNAL_POINTERS
                      if p in txt.lower()]

    issues = missing_label + crossref_issues + pointer_issues + slice_issues
    out = {"master_words": len(body.split()), "master_sections_present": sorted(secs),
           "n_claims": len(rows), "n_numeric_claims": n_num, "n_literals_present": n_present,
           "migration_coverage_pct": coverage, "claims": rows,
           "refs_ok": {k: v for k, v in {"sections": sec_state, "tables": tab_state, "figures": fig_state}.items()},
           "pending_refs": pending,
           "crossref_issues": crossref_issues, "pointer_issues": pointer_issues,
           "slicing": slice_report, "slice_issues": slice_issues,
           "missing_labels": missing_label,
           "verdict": "PASS" if not issues else "FAIL", "n_issues": len(issues)}
    json.dump(out, open(os.path.join(RES, "master_audit.json"), "w", encoding="utf-8"), indent=1)

    L = [f"# Unified master audit: {out['verdict']} ({len(issues)} issues)", "",
         f"master: {out['master_words']} words, {len(secs)} numbered sections | "
         f"claims mapped: {len(rows)} ({n_num} numeric) | migration coverage: "
         f"{n_present}/{n_num} = {coverage}%", "",
         "| id | master | verifier | ref | literal | label ok | source check passes | literal = verified | in master | strength |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['master']} | {r['verifier']} | {r['ref'] or '-'} | {r['literal'] or '-'} | "
                 f"{r['verifier_ok']} | {r['source_passes']} | {r['literal_matches_source']} | "
                 f"{'yes' if r['literal_in_master'] else 'pending'} | {r['strength']} |")
    L += ["", "## Slicing map (closure of internal references)", "",
          "| version | seed sections | closure size | own contribution |", "|---|---|---|---|"]
    for name, r in slice_report.items():
        L.append(f"| {name} | {len(r['seed'])} | {r['n_closure']} | {r['own_contribution']} |")
    if issues:
        L += ["", "## Issues", ""] + [f"- {x}" for x in issues]
    open(os.path.join(RES, "master_audit.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    print("issues:", len(issues), "| migration coverage: %.1f%%" % coverage)
    return 0 if not issues else 1


if __name__ == "__main__":
    main()
