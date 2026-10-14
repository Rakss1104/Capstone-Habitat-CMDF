"""Conclusion text, generated from result objects and the pre-specified decision rule (PROTOCOL.md section 5)."""
from __future__ import annotations


def _auc(summary, s, m):
    r = summary.query("set == @s and model == @m")
    if not len(r):
        return "not evaluated"
    r = r.iloc[0]
    return f"{r['_auc']:.3f} ({r['_lo']:.3f}–{r['_hi']:.3f})"


def conclusion_text(study):
    R = study.R
    S, prim, pn = R["summary"], R["primary"], R["pairing_null"]
    supported = prim["_nb_lo"] > 0 and pn["p"] < 0.05
    L = ["# CMDF study: conclusion", ""]
    L.append(f"Protocol: `PROTOCOL.md` (SHA-256 {study.protocol_sha[:16]}…). Cohort: UTSW {len(study.ids['UTSW'])} patients "
             f"({int(study.base.loc[study.ids['UTSW'], 'label'].sum())} codeleted), UCSF {len(study.ids['UCSF'])} "
             f"({int(study.base.loc[study.ids['UCSF'], 'label'].sum())}); repeated 10 × 5-fold CV, folds identical to the habitat study.")
    L += ["", "## Primary hypothesis (H1: P+D > P within UTSW)", ""]
    L.append(f"- AUROC P = {_auc(S, 'UTSW', 'P')}; P+D = {_auc(S, 'UTSW', 'P+D')}.")
    L.append(f"- ΔAUROC (fold mean) = {prim['ΔAUROC (fold mean)']:.3f}, Nadeau–Bengio 95% CI {prim['NB 95% CI']}, "
             f"p = {prim['NB p']:.3f}; folds with Δ > 0: {prim['folds with Δ>0']}.")
    L.append(f"- Pairing null (co-location shuffled, {pn['n']} replicates): observed Δ (repeat 1) {pn['delta_obs_rep1']:.3f}, "
             f"null mean {pn['null_mean']:.3f}, p = {pn['p']:.3f}.")
    L.append(f"- By the pre-specified rule, H1 is **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**.")
    L += ["", "## Secondary comparisons (Benjamini–Hochberg across S1–S7)", ""]
    for _, r in R["secondary"].iterrows():
        L.append(f"- {r['id']} {r['set']}: {r['comparison']} = {r['ΔAUROC (fold mean)']:.3f} (NB 95% CI {r['NB 95% CI']}; "
                 f"BH-adjusted p = {r['NB p (BH-adjusted)']:.3f}).")
    L += ["", "## AUROC by model (repeat-averaged out-of-fold, 95% bootstrap CI)", "",
          "| set | model | n | codeleted | AUROC |", "|---|---|---|---|---|"]
    for _, r in S.iterrows():
        L.append(f"| {r['set']} | {r['model']} | {r['n']} | {r['n_codeleted']} | {r['AUROC (repeat-avg OOF, 95% CI)']} |")
    h2 = R["H2"]
    L += ["", "## H2: T2–FLAIR mismatch fraction lower in codeleted tumours", ""]
    for _, r in h2.iterrows():
        delta = r["Cliff's delta (>0: higher if codeleted)"]
        dirn = "lower" if delta < 0 else "higher"
        L.append(f"- {r['set']}: median {r['median codeleted']:.3f} (codeleted, n={r['n_codel']}) vs {r['median non-codeleted']:.3f} "
                 f"(non-codeleted, n={r['n_non']}); Cliff's δ = {delta:.3f} "
                 f"(95% CI {r['95% CI']}), p = {r['p']:.4f} — {dirn} in codeleted.")
    lp = R["label_permutation"]
    L += ["", "## Null tests and transport", ""]
    for _, r in lp.iterrows():
        L.append(f"- Label permutation, UTSW {r['model']}: observed {r['observed AUROC (cohort-stratified folds)']:.3f} vs null mean "
                 f"{r['null mean']:.3f} (95th pct {r['null 95th pct']:.3f}), p = {r['p (one-sided)']:.3f}.")
    for _, r in R["loco"].iterrows():
        if r["status"] == "run":
            L.append(f"- Leave-one-cohort-out {r['train']} → {r['test']}, {r['model']}: AUROC {r['AUROC (95% CI)']}.")
        else:
            L.append(f"- Leave-one-cohort-out {r['train']} → {r['test']}: {r['status']}.")
    L += ["", "## Robustness", ""]
    for _, r in R["robustness"].iterrows():
        L.append(f"- {r['variant']}: Δ(P+D − P) = {r['ΔAUROC (fold mean)']:.3f} (NB 95% CI {r['NB 95% CI']}).")
    for _, r in R["ablation"].iterrows():
        L.append(f"- Ablation {r['variant']}: AUROC {r['AUROC (repeat-avg OOF)']:.3f} (Δ {r['ΔAUROC vs full P+D']:+.3f}).")
    L += ["", "## Methodological arm: the original notebooks' protocol on the same UTSW P+D features", ""]
    for _, r in R["naive"].iterrows():
        L.append(f"- {r['variant']}: best classifier {r['best_classifier']} AUROC {r['best_auc']:.3f} ± {r['best_sd']:.3f}; "
                 f"the same protocol on permuted labels gives {r['null_mean']:.3f} on average (95th pct {r['null_95']:.3f}, "
                 f"max {r['null_max']:.3f}), p = {r['p']:.3f}. Leakage-free estimate: {r['rigorous P+D AUROC (repeat-avg OOF)']:.3f}.")
    L += ["", "## Deviations from protocol", ""]
    L += [f"- {d}" for d in R["deviations"]] or ["- None."]
    L += ["", "## Limitations", "",
          "- Single-centre primary analysis (UTSW, 36 events); UCSF has 8 codeleted patients, so transport is descriptive.",
          "- No untouched external cohort.",
          "- Whole-abnormality masks; the mismatch features do not separate tumour core from oedema."]
    return "\n".join(L) + "\n"
