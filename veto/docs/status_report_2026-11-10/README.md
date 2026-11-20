# Status-quo report of the veto background study (2026-11-10)

LaTeX article, Overleaf-ready (pdflatex + bibtex). Compile: `latexmk -pdf main.tex` (or upload the folder to Overleaf).

- `main.tex` preamble (style of `docs/issues_report/veto_issues.tex`), `sections/01..09_*.tex`, `refs.bib`,
  `figures/` (copies of existing plots), `logos/`.
- `main.pdf` is the build (gitignored: `*.pdf`).
- Sources of every number: `docs/STATUS.md`, `ISSUES.txt` (A1-D16), `CHANGELOG.md`, `docs/neutrinos_2026.md`,
  `docs/shield_iron_dis.md`, `docs/full_sim_inventory.md`, `docs/v0_production.md`; each table / figure carries its
  source path. No simulation was run for this report.
- Figures: `v0_vs_W.pdf`, `v0_ms7_fate.pdf` (docs/figures/v0_production), `dis_sigma_generators.pdf`,
  `kl_q2w_A_geant4_11.2.pdf` (results/shield_dis), `dis_chain_overview.pdf` (docs/dis_chain_pdf/overview.pdf),
  `d5_summary.png` (results/d5_convention), `em_debris.png`, `complete_validation.png`
  (/disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/results/...).
- To push to the Overleaf GitHub repo: copy this folder as the repo root (main.tex at the top).
