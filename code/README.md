# Replication workspace

Working files are grouped by paper:

- `hartnett_saleh/replication.ipynb`: working Tables 1–2 replication/audit notebook.
- `hartnett_saleh/replication_package/`: extracted authors' Stata package.
- `hartnett_saleh/audit_outputs/`: working notebook exports.
- `cremaschi_etal/replication.ipynb`: working Cremaschi notebook. Authors' package
  and data remain under `../nov_2025/cremaschi_etal/`.
- `clark_golder_2006/replication.ipynb`: working Clark–Golder notebook. Its expected
  input directory is repository-root `dataverse_files/`, which is currently absent.

`fragility_audits/` is shared infrastructure: protocol, tests, master datasets,
staging, and paper archives. Hartnett–Saleh's completed audit scripts and results
are in `fragility_audits/papers/hartnett_saleh_2025/`; see its `MAIN_TEXT_AUDIT.md`
for reproduction commands. Archived notebooks preserve historical snapshots;
the working notebooks above use repository-relative paths and can be launched
from the repository root or a directory within it.

The Hartnett–Saleh working notebook's archival cells cover Tables 1–2 only.
After rerunning those cells, run the main-text extension/finalizer, master builder,
and comparison generator to restore Tables 1–4 coverage. Moving files did not
rerun any perturbation experiment or alter its numerical results.
