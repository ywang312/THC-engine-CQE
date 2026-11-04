"""Generator-rank study and compressed CQE versus CNOT estimates (milestone 1).

For each system this script

1. checks the exact first-step relation K_1 = 2 rank(4V - 2X) against the
   Cholesky rank of the (ov|ov) block (diagnostics.generator_rank);
2. follows the exact steepest-descent CQE and profiles every generator: the
   leaf rank, the leaves needed for 90/99/99.9% of ||g||^2, and the
   symmetry-unique elements and spin-orbital excitations needed in the
   canonical, natural and two Boys-localized bases, each with CNOT estimates
   (diagnostics.cost). 'localized-ov' localizes the occupied and the virtual
   orbitals separately, which keeps the HF determinant and the
   occupied/virtual block structure of the generator. 'localized' localizes
   the whole active space and therefore mixes occupied and virtual orbitals,
   which destroys that block structure;
3. runs compressed CQE with 'elements' (canonical and both Boys bases),
   'leaves' and 'leaves-trotter' (first-order product) for several budgets per
   iteration. It then reports the energy error against the cumulative CNOT
   estimate, as the best error reached within matched CNOT budgets.

Outputs: <out>/generator_rank_<sys>.json and <out>/generator_rank_<sys>.md for
each system, plus <out>/generator_rank_summary.md for all systems of the run.
The summary starts with a headline table (:func:`headline_markdown`) computed
from the results, so quoted ratios can be checked against it.

Every CNOT number is an ESTIMATE from diagnostics.cost; see its docstring for
the assumptions and sources.

Usage:
    python benchmarks/run_generator_rank.py --systems h4,h6 --max-iter 15 --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

CHEMICAL_ACCURACY = 1.6e-3  # Hartree (1 kcal/mol)
LOCALIZED_BASES = ("localized-ov", "localized")
HEADLINE_ITERS = range(1, 6)  # rank-study iterations of the headline medians


def build_system(name: str):
    from thc_engine_cqe import systems

    table = {
        "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
        "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
        "h6s": lambda: systems.hydrogen_chain(6, 3.0, "bohr"),
        "h8": lambda: systems.hydrogen_chain(8, 1.8, "bohr"),
        "lih": lambda: systems.lih(),
        "n2": lambda: systems.n2(1.098),
        "n2s": lambda: systems.n2(2.0),
    }
    if name not in table:
        raise SystemExit(f"unknown system {name!r}; choose from {sorted(table)}")
    return table[name]()


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def run_system(name: str, args) -> dict:
    from thc_engine_cqe.diagnostics import generator_rank as gr
    from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian

    t0 = time.time()
    act = build_system(name)
    e_fci = float(act.fci()[0])
    e_hf = float(Hamiltonian(act).energy(FCISpace.of(act).hf()))
    out = {"system": name, "label": act.label, "norb": act.norb, "nelec": list(act.nelec),
           "e_fci": e_fci, "e_hf": e_hf, "settings": vars(args).copy()}
    out["settings"].pop("func", None)
    out["relation"] = gr.first_step_rank_relation(act)
    print(f"[{name}] {act.label}: N={act.norb}, nelec={act.nelec}, E_HF-E_FCI={e_hf - e_fci:.3e}", flush=True)
    t = time.time()
    out["study"] = gr.rank_study(act, max_iter=args.max_iter, e_fci=e_fci)
    print(f"[{name}] rank study: {len(out['study'])} iterations in {time.time() - t:.1f} s", flush=True)

    runs = []
    plans = [("elements", p, None, "canonical") for p in args.elements_per]
    if args.localized_elements:
        bases = gr.orbital_bases(act)  # 'localized-ov' and 'localized' when mol/mo_coeff exist
        for label in LOCALIZED_BASES:
            if label in bases:
                plans += [("elements", p, bases[label], label) for p in args.elements_per]
    plans += [("leaves", p, None, "-") for p in args.leaves_per]
    if not args.no_trotter:
        plans += [("leaves-trotter", p, None, "-") for p in args.leaves_per]
    for scheme, per, basis, label in plans:
        t = time.time()
        res = gr.compressed_cqe(act, scheme, per, max_iter=args.cqe_iter, e_fci=e_fci,
                                basis=basis, basis_label=label if scheme == "elements" else None)
        d = res.to_dict()
        d["errors"] = [float(x) for x in res.errors]
        d["cum_cnots_all"] = [int(x) for x in res.cum_cnots("all")]
        d["cum_cnots_linear"] = [int(x) for x in res.cum_cnots("linear")]
        d["seconds"] = time.time() - t
        runs.append(d)
        print(f"[{name}] {scheme:15s} {label:12s} per={per:<3d} final error {d['errors'][-1]:.2e} "
              f"cum CNOT {d['cum_cnots_all'][-1]:>7d} ({d['seconds']:.1f} s)", flush=True)
    out["runs"] = runs
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------- markdown
def _fmt_e(x: float) -> str:
    return f"{x:.2e}"


def _run_name(r: dict) -> str:
    if r["scheme"] == "elements":
        return f"elements[{r['basis']}]"
    return r["scheme"]


def best_within_budget(run: dict, budget: float, conn: str) -> float:
    cum = run[f"cum_cnots_{conn}"]
    errs = run["errors"]
    ok = [e for c, e in zip(cum, errs) if c <= budget]
    return min(ok) if ok else float("nan")


def cnots_to_reach(run: dict, threshold: float, conn: str) -> int | None:
    for c, e in zip(run[f"cum_cnots_{conn}"], run["errors"]):
        if e < threshold:
            return int(c)
    return None


def budgets_for(runs: list[dict], conn: str) -> list[int]:
    top = max(max(r[f"cum_cnots_{conn}"]) for r in runs)
    b, out = 250, []
    while b < 2 * top:
        out.append(b)
        b *= 2
    return out


def markdown_system(res: dict) -> str:
    lines = []
    name, rel, study = res["system"], res["relation"], res["study"]
    lines.append(f"## {name}: {res['label']}")
    lines.append("")
    lines.append(f"N = {res['norb']} spatial orbitals ({2 * res['norb']} qubits), nelec = {tuple(res['nelec'])}, "
                 f"E_HF - E_FCI = {res['e_hf'] - res['e_fci']:.4e} Ha.")
    lines.append("")
    lines.append("**First step from RHF.** The first generator has exactly K_1 = 2 rank(W) leaves, "
                 "where W = 4(ia|jb) - 2(ib|ja). The Cholesky rank r_ov of (ia|jb) does not bound it "
                 "on its own:")
    lines.append("")
    lines.append("| K_1 | 2 rank W | r_ov | rank X | n_occ n_vir | bound 2 min(n_ov, r_ov + rank X) | max abs(A_ovov - W) |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| {rel['K1']} | {2 * rel['rank_W']} | {rel['r_ov']} | {rel['rank_X']} | {rel['n_ov']} | "
                 f"{rel['bound']} | {rel['max_dev']:.1e} |")
    lines.append("")
    fr = ("0.9", "0.99", "0.999")
    lines.append("**Leaves along the exact CQE.** Leaf counts are basis invariant. Each count is the "
                 "number of leaves for 90 / 99 / 99.9% of ||g||^2, rounded up to complete "
                 "conjugate pairs. CNOTs are for one step at 99%.")
    lines.append("")
    lines.append("| it | E - E_FCI | norm A | leaf rank | leaves 90/99/99.9 | CNOT@99 all | CNOT@99 linear |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in study:
        lp = r["leaf_profile"]
        nl = "/".join(str(lp["n_leaves_complete"][f]) for f in fr)
        lines.append(f"| {r['iteration']} | {_fmt_e(r['error'])} | {r['residual_norm']:.3f} | {lp['rank']} | {nl} | "
                     f"{lp['cnots']['all']['0.99']} | {lp['cnots']['linear']['0.99']} |")
    lines.append("")
    bases = list(study[0]["element_profile"])
    lines.append("**Elements along the exact CQE.** Element counts depend on the basis. Each cell "
                 "gives the symmetry-unique spin-free elements for 90 / 99 / 99.9% of ||g||^2, "
                 "then [spin-orbital excitations at 99%], then CNOTs for one step at 99% "
                 "(all-to-all). 'natural' is the natural-orbital basis of the current state. "
                 "'localized-ov' is Boys localization within the occupied and within the virtual "
                 "orbitals, which keeps the HF determinant and the occupied/virtual block structure. "
                 "'localized' is Boys over the whole active space. It mixes occupied and virtual "
                 "orbitals and so destroys that block structure.")
    lines.append("")
    lines.append("| it | " + " | ".join(bases) + " |")
    lines.append("|---|" + "---|" * len(bases))
    for r in study:
        cells = []
        for b in bases:
            ep = r["element_profile"][b]
            ne = "/".join(str(ep["n_elements"][f]) for f in fr)
            cells.append(f"{ne} [{ep['n_excitations']['0.99']}] {ep['cnots']['all']['0.99']}")
        lines.append(f"| {r['iteration']} | " + " | ".join(cells) + " |")
    lines.append("")
    runs = res["runs"]
    if not runs:
        return "\n".join(lines)
    lines.append("**Compressed CQE.** Each run starts from HF and keeps `per` elements or leaves per "
                 "iteration. 'elements' and 'leaves' use the exact exponential. 'leaves-trotter' "
                 "uses the first-order product of the kept leaves (the circuit that the leaf "
                 "CNOT estimate counts). The basis of each 'elements' run is in brackets. CNOTs "
                 "are cumulative over the whole circuit.")
    lines.append("")
    lines.append("| run | per | iters | final E - E_FCI | cum CNOT all | cum CNOT linear | "
                 "CNOT to 1.6e-3 (all) | mean captured descent |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        reach = cnots_to_reach(r, CHEMICAL_ACCURACY, "all")
        capt = np.mean([x["captured"] for x in r["records"]]) if r["records"] else float("nan")
        lines.append(f"| {_run_name(r)} | {r['per_iter']} | {len(r['records'])} | {_fmt_e(r['errors'][-1])} | "
                     f"{r['cum_cnots_all'][-1]} | {r['cum_cnots_linear'][-1]} | "
                     f"{reach if reach is not None else '-'} | {capt:.3f} |")
    lines.append("")
    for conn in ("all", "linear"):
        lines.append(f"**Matched budgets ({conn} connectivity).** Each cell is the best E - E_FCI "
                     "reached with cumulative CNOTs at or below the budget, minimized over `per`, "
                     "with the best `per` in parentheses.")
        lines.append("")
        names = []
        for r in runs:
            if _run_name(r) not in names:
                names.append(_run_name(r))
        lines.append("| CNOT budget | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for b in budgets_for(runs, conn):
            cells = []
            for nm in names:
                cand = [(best_within_budget(r, b, conn), r["per_iter"]) for r in runs if _run_name(r) == nm]
                cand = [c for c in cand if not np.isnan(c[0])]
                if not cand:
                    cells.append("-")
                    continue
                e, p = min(cand)
                cells.append(f"{_fmt_e(e)} ({p})")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def _best_reach(runs: list[dict], name: str, conn: str = "all") -> int | None:
    """Fewest cumulative CNOTs to reach chemical accuracy, minimized over `per` (None if never)."""
    reach = [cnots_to_reach(r, CHEMICAL_ACCURACY, conn) for r in runs if _run_name(r) == name]
    reach = [x for x in reach if x is not None]
    return min(reach) if reach else None


def headline(res: dict) -> dict:
    """Headline numbers of one system, computed from its results.

    one_step[name]: median over rank-study iterations HEADLINE_ITERS of the
    one-step CNOT estimate (all-to-all) at 99% of ||g||^2, for 'leaves' and
    for 'elements[<basis>]' in every basis of the rank study.
    reach[name]: fewest cumulative CNOTs (all-to-all) with which a compressed
    run gets below chemical accuracy, minimized over `per` (None if no run does).
    """
    study = [r for r in res["study"] if r["iteration"] in HEADLINE_ITERS]
    one = {"leaves": _median([r["leaf_profile"]["cnots"]["all"]["0.99"] for r in study])}
    for b in (res["study"][0]["element_profile"] if res["study"] else {}):
        if b == "natural":
            continue
        one[f"elements[{b}]"] = _median([r["element_profile"][b]["cnots"]["all"]["0.99"] for r in study])
    names = []
    for r in res["runs"]:
        if _run_name(r) not in names:
            names.append(_run_name(r))
    return {"one_step": one, "reach": {nm: _best_reach(res["runs"], nm) for nm in names}}


def _ratio(a, b) -> str:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return "-"
    return f"{a / b:.2f}"


def headline_markdown(results: list[dict]) -> str:
    """Per-system leaves vs elements comparison (computed, not hand-written)."""
    if not results:
        return ""
    heads = [headline(res) for res in results]
    one_names = []
    reach_names = []
    for h in heads:
        one_names += [k for k in h["one_step"] if k not in one_names]
        reach_names += [k for k in h["reach"] if k not in reach_names]
    it = f"{HEADLINE_ITERS.start}-{HEADLINE_ITERS.stop - 1}"
    lines = ["## Headline: leaves vs elements (computed from the results below)", ""]
    lines.append(f"**One step at 99% of ||g||^2.** Median over rank-study iterations {it} of the one-step "
                 "CNOT estimate (all-to-all). The last column is leaves / elements[canonical]; above 1 "
                 "means elementwise compilation in canonical orbitals is cheaper.")
    lines.append("")
    lines.append("| system | " + " | ".join(one_names) + " | leaves / elements[canonical] |")
    lines.append("|---|" + "---|" * (len(one_names) + 1))
    for res, h in zip(results, heads):
        o = h["one_step"]
        cells = [f"{o[k]:.0f}" if k in o and np.isfinite(o[k]) else "-" for k in one_names]
        lines.append(f"| {res['system']} | " + " | ".join(cells) + " | "
                     f"{_ratio(o.get('leaves'), o.get('elements[canonical]'))} |")
    lines.append("")
    if any(h["reach"] for h in heads):
        lines.append(f"**CNOTs to chemical accuracy ({CHEMICAL_ACCURACY:g} Ha).** Fewest cumulative CNOTs "
                     "(all-to-all) of a compressed run below chemical accuracy, minimized over `per`; "
                     "'-' if no run gets there. The last column is leaves / elements[canonical].")
        lines.append("")
        lines.append("| system | " + " | ".join(reach_names) + " | leaves / elements[canonical] |")
        lines.append("|---|" + "---|" * (len(reach_names) + 1))
        for res, h in zip(results, heads):
            rc = h["reach"]
            cells = [str(rc[k]) if rc.get(k) is not None else "-" for k in reach_names]
            lines.append(f"| {res['system']} | " + " | ".join(cells) + " | "
                         f"{_ratio(rc.get('leaves'), rc.get('elements[canonical]'))} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="h4,h6,lih",
                    help="comma-separated subset of h4,h6,h6s,lih,n2,n2s,h8 (default h4,h6,lih)")
    ap.add_argument("--max-iter", type=int, default=15, help="iterations of the exact rank study")
    ap.add_argument("--cqe-iter", type=int, default=20, help="iterations of each compressed CQE run")
    ap.add_argument("--elements-per", type=_ints, default=[2, 4, 8, 16],
                    help="elements kept per iteration (comma-separated)")
    ap.add_argument("--leaves-per", type=_ints, default=[2, 4, 8],
                    help="leaves kept per iteration, rounded up to complete pairs (comma-separated)")
    ap.add_argument("--no-trotter", action="store_true", help="skip the 'leaves-trotter' runs")
    ap.add_argument("--localized-elements", action=argparse.BooleanOptionalAction, default=True,
                    help="also run 'elements' in the Boys-localized bases 'localized-ov' "
                         "(occupied and virtual separately) and 'localized' (whole active space)")
    ap.add_argument("--threads", type=int, default=None, help="PySCF OpenMP threads")
    ap.add_argument("--out", default="results/", help="output directory")
    args = ap.parse_args(argv)
    if args.threads is not None:
        from pyscf import lib

        lib.num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    sections = []
    results = []
    for name in names:
        res = run_system(name, args)
        results.append(res)
        with open(os.path.join(args.out, f"generator_rank_{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        md = markdown_system(res)
        with open(os.path.join(args.out, f"generator_rank_{name}.md"), "w") as fh:
            fh.write(md + "\n")
        sections.append(md)
        print(f"[{name}] done in {res['seconds']:.1f} s", flush=True)
    header = ["# Generator rank and compressed CQE (estimates)", "",
              "CNOT counts are estimates from `thc_engine_cqe.diagnostics.cost`, which "
              "states the assumptions and sources (Yordanov et al. 2020 for the 13-CNOT "
              "double excitation; Kivlichan et al. 2018 and Motta et al. 2021 for Givens "
              "networks and diagonal-Coulomb layers). E_FCI is the exact active-space energy.", "",
              headline_markdown(results)]
    with open(os.path.join(args.out, "generator_rank_summary.md"), "w") as fh:
        fh.write("\n".join(header + sections) + "\n")
    print(f"wrote {args.out}/generator_rank_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
