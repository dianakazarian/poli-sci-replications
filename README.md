# Political science replication audits

Code and methods for baseline replication and Gaussian sensitivity audits.
See [the protocol](code/fragility_audits/protocols/gaussian_protocol_v1.md)
and [workspace guide](code/README.md).

This repository publishes selected scripts and methods documentation only.
Source datasets, authors' replication packages, PDFs, notebooks, empirical
results, registries, manifests, and numerical reference tests are private local
artifacts and excluded through a default-deny `.gitignore`.

## Environment and use

Python scripts require numpy, pandas, scipy, and patsy. Independent regression
validation also uses statsmodels. No dependency versions are pinned here;
use the private audit manifests for the recorded research environment.

The scripts expect locally supplied replication packages and existing audit
archives. This public code snapshot is not a standalone reproduction package:
Tables 1–2 currently depend on a private notebook and the finalizer requires
private claim rules and baseline artifacts. No private inputs are downloaded
by these scripts.

Run the public synthetic aggregation check from the repository root:

```sh
python3 code/fragility_audits/tests/test_summary_public.py
```

The test uses invented inputs and contains no empirical findings.
