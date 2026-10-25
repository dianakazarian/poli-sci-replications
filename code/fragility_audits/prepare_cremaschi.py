"""Stage existing Cremaschi evidence without admitting it to gaussian_v1 masters."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = REPO / 'nov_2025/cremaschi_etal'
AUDIT = SOURCE / 'cremaschietal-dataverse/independent_audit'
DEST = ROOT / 'staging/cremaschi_etal'


def main():
    for folder in ('original', 'original_exports', 'code', 'registries', 'results'):
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    for source, target in (
        (SOURCE / 'without-roots-the-political-consequences-of-collective-economic-shocks.pdf', 'original/article.pdf'),
        (SOURCE / 'cremaschi_etal_data.zip', 'original/replication.zip'),
        (SOURCE / 'cremaschietal-dataverse/README.md', 'original/README.md'),
        (REPO / 'code/cremaschi_etal/replication.ipynb', 'code/audit.ipynb'),
    ):
        shutil.copy2(source, DEST / target)
    for source in sorted(AUDIT.iterdir()):
        if source.is_file():
            shutil.copy2(source, DEST / 'original_exports' / source.name)
    notebook = json.loads((DEST / 'code/audit.ipynb').read_text())
    # Preserve executable R strings with their original cell numbers; these are
    # archival extracts, and some depend on designs produced by earlier cells.
    import ast
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and node.value.s.lstrip().startswith(('library(', 'set.seed(', 'if (!requireNamespace')):
                        (DEST / 'code' / ('cell_%02d_%s.R' % (i, target.id))).write_text(node.value.s)
    with (AUDIT / 'frozen_claims.csv').open() as f:
        frozen = list(csv.DictReader(f))
    with (DEST / 'registries/baseline_targets.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(frozen[0]))
        writer.writeheader()
        writer.writerows(frozen)
    summaries = []
    for filename, design in (
        ('table1_farright_1000_draws.csv', 'fixed_count_infection_omission'),
        ('probabilistic_omission_draws.csv', 'bernoulli_infection_omission'),
        ('table1_gaussian_inference_pilot.csv', 'gaussian_pilot_requires_scaling_review'),
    ):
        with (AUDIT / filename).open() as f:
            groups = {}
            for row in csv.DictReader(f):
                groups.setdefault(float(row['delta']), []).append(row)
        for delta, rows in sorted(groups.items()):
            summaries.append(dict(design=design, delta=delta, n_simulations=len(rows),
                S_sign=sum(float(r['beta']) > 0 for r in rows)/len(rows),
                S_inferential=sum(float(r['p_value']) < .05 for r in rows)/len(rows),
                S_claim=sum(float(r['beta']) > 0 and float(r['p_value']) < .05 for r in rows)/len(rows),
                source=filename, master_eligible=False))
    with (DEST / 'results/existing_experiment_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    files = [dict(path=str(p.relative_to(DEST)), size_bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'preparation_manifest.json']
    manifest = dict(archive_status='staged_not_master_eligible', protocol_version=None,
        source_notebook='code/cremaschi_etal/replication.ipynb',
        publication_year=None, doi=None,
        note='Bibliographic fields and persistent paper/model IDs require verification before promotion.',
        files=files)
    (DEST / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Staged %s files; %s experiment summaries. Master datasets untouched.' % (len(files), len(summaries)))


if __name__ == '__main__':
    main()
