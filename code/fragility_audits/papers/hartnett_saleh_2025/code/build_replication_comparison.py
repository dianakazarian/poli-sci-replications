"""Compare every reported main-text slope against the archived replication.
Uses authors' TeX tables and the baseline registry; does not rerun regressions.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

ARCHIVE = Path(__file__).resolve().parent.parent
PACKAGE = ARCHIVE.parents[3] / 'code/hartnett_saleh/replication_package/APSR Replication Package'
LABELS = {
    'Cotton': 'cotton_post', 'Cereals': 'cereal_post',
    'Cotton $\\times$ Democratic Speeches Per MP': 'cotton_speech_post',
    'Cotton $\\times$ N. Urabi V. Headmen Arrests': 'cotton_village_post',
    'Cotton $\\times$ Large Estates Land Share (Q3)': 'cotton_estate_q3_post',
    'Cotton $\\times$ Large Estates Land Share (Q4)': 'cotton_estate_q4_post',
    'Cotton $\\times$ Prop. Slaves': 'cotton_slave_post',
    'Prop. Democratic Speeches': 'speech_post',
    'N. Urabi V. Headmen Arrests': 'village_post',
    'Large Estates Land Share (Q1)': 'estate_q1_post',
    'Large Estates Land Share (Q3)': 'estate_q3_post',
    'Large Estates Land Share (Q4)': 'estate_q4_post',
    'Prop. Slaves': 'slave_post',
}


def number(cell):
    match = re.search(r'-?\d+\.\d+', cell)
    return float(match.group()) if match else None


def main():
    with (ARCHIVE / 'registries/coefficient_registry.csv').open() as f:
        registry = {(r['model_id'], r['variable']): r for r in csv.DictReader(f)}
    with (ARCHIVE / 'registries/model_registry.csv').open() as f:
        models = {r['model_id']: r for r in csv.DictReader(f)}
    rows = []
    for table in ['1', '2', '3a', '3b', '4a', '4b']:
        source = PACKAGE / 'Results' / ('Table' + table + '.tex')
        lines = source.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.startswith('Post-1882'):
                continue
            label = line.split('&')[0].split('$\\times$ ', 1)[1]
            variable = LABELS[label]
            betas = line.split('&')[1:]
            ses = lines[i+1].split('&')[1:]
            for column, (bcell, scell) in enumerate(zip(betas, ses), 1):
                published_beta, published_se = number(bcell), number(scell)
                if published_beta is None:
                    assert published_se is None
                    continue
                mid = 'table' + table + '_col' + str(column)
                ref = registry[(mid, variable)]
                beta, se = float(ref['baseline_beta']), float(ref['baseline_se'])
                beta_match = abs(beta-published_beta) <= .0005 + 1e-12
                se_match = abs(se-published_se) <= .0005 + 1e-12
                rows.append(dict(paper_id=ref['paper_id'], table=int(table[0]),
                    panel=table[1:] or '', column=column, model_id=mid,
                    outcome=models[mid]['outcome'], variable=variable,
                    published_label=label, published_beta=published_beta,
                    replicated_beta=beta, beta_difference=beta-published_beta,
                    published_se=published_se, replicated_se=se,
                    se_difference=se-published_se, published_decimal_places=3,
                    beta_matches_rounding=beta_match, se_matches_rounding=se_match,
                    replication_status='matches_published_rounding' if beta_match and se_match else 'discrepancy',
                    replicated_p=float(ref['baseline_p']),
                    n_observations=int(ref['n_observations']), n_clusters=int(ref['n_clusters']),
                    published_source=str(source.relative_to(ARCHIVE.parents[3]))))
    keys = [(r['model_id'], r['variable']) for r in rows]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(registry), 'Published/registry coefficient coverage differs'
    assert len(rows) == 95
    output = ARCHIVE / 'results/published_replication_comparison.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest_path = ARCHIVE / 'results/audit_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'] = [dict(path=str(p.relative_to(ARCHIVE)), size_bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(ARCHIVE.rglob('*')) if p.is_file() and p != manifest_path and '__pycache__' not in p.parts]
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('%s: %d comparisons, %d match, %d discrepancies' %
          (output.name, len(rows), sum(r['replication_status']=='matches_published_rounding' for r in rows),
           sum(r['replication_status']=='discrepancy' for r in rows)))


if __name__ == '__main__':
    main()
