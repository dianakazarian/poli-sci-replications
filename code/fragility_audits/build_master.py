from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"
MASTER = ROOT / "master"
PROTOCOL = "gaussian_v1"

MASTER.mkdir(parents=True, exist_ok=True)


def load_archives():
    paper_rows = []
    model_frames = []
    coefficient_frames = []
    claim_frames = []

    manifests = sorted(
        PAPERS.glob("*/results/audit_manifest.json")
    )

    if not manifests:
        raise RuntimeError("No audit archives found.")

    for manifest_path in manifests:
        paper_dir = manifest_path.parent.parent

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest["protocol_version"] != PROTOCOL:
            raise ValueError(
                "Incompatible protocol: " + manifest["paper_id"]
            )

        if not str(manifest.get("archive_status", "")).startswith(
            "complete"
        ):
            raise ValueError(
                "Incomplete archive: " + manifest["paper_id"]
            )

        with open(
            paper_dir / "registries/paper_metadata.json"
        ) as f:
            metadata = json.load(f)

        models = pd.read_csv(
            paper_dir / "registries/model_registry.csv"
        )
        coefficients = pd.read_csv(
            paper_dir / "results/coefficient_stability.csv"
        )
        claims = pd.read_csv(
            paper_dir / "results/claim_stability.csv"
        )

        required = {
            "models": (
                models,
                ["paper_id", "model_id", "outcome",
                 "n_observations", "n_clusters"]
            ),
            "coefficients": (
                coefficients,
                ["paper_id", "model_id", "variable",
                 "perturbation_design", "delta",
                 "S_sign", "S_inferential",
                 "inferential_failure_rate",
                 "significant_5pct"]
            ),
            "claims": (
                claims,
                ["paper_id", "claim_id",
                 "perturbation_design", "delta", "S_claim"]
            )
        }

        for name, (frame, columns) in required.items():
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(
                    "{}: {} missing {}".format(
                        manifest["paper_id"], name, sorted(missing)
                    )
                )
            if not frame["paper_id"].eq(
                manifest["paper_id"]
            ).all():
                raise ValueError(
                    "Paper ID mismatch: " + name
                )

        expected = manifest["counts"]

        assert len(models) == expected["models"]
        assert len(coefficients) == expected[
            "coefficient_stability_records"
        ]
        assert len(claims) == expected[
            "claim_stability_records"
        ]

        paper_rows.append({
            "paper_id": manifest["paper_id"],
            "authors": "; ".join(metadata["authors"]),
            "title": metadata["title"],
            "journal": metadata["journal"],
            "publication_year": metadata["publication_year"],
            "doi": metadata["doi"],
            "audit_scope": manifest["audit_scope"],
            "protocol_version": manifest["protocol_version"],
            "archive_status": manifest["archive_status"],
            "n_models": len(models),
            "n_coefficient_records": len(coefficients),
            "n_claim_records": len(claims)
        })

        model_frames.append(models)
        coefficient_frames.append(coefficients)
        claim_frames.append(claims)

    datasets = {
        "papers": pd.DataFrame(paper_rows),
        "models": pd.concat(model_frames, ignore_index=True),
        "coefficients": pd.concat(
            coefficient_frames, ignore_index=True
        ),
        "claims": pd.concat(claim_frames, ignore_index=True)
    }

    keys = {
        "papers": ["paper_id"],
        "models": ["paper_id", "model_id"],
        "coefficients": [
            "paper_id", "model_id", "variable",
            "perturbation_design", "delta"
        ],
        "claims": [
            "paper_id", "claim_id",
            "perturbation_design", "delta"
        ]
    }

    for name, frame in datasets.items():
        if frame.duplicated(subset=keys[name]).any():
            raise ValueError("Duplicate records: " + name)

    return datasets


def summarize(coefficients, claims):
    # Failure rates for originally significant targets only
    significant = coefficients[
        coefficients["significant_5pct"].fillna(False)
    ].copy()

    by_coefficient = (
        significant
        .groupby(["paper_id", "perturbation_design", "delta"])
        .agg(
            n_significant_coefficients=("model_id", "size"),
            mean_inferential_failure=(
                "inferential_failure_rate", "mean"
            ),
            median_inferential_failure=(
                "inferential_failure_rate", "median"
            ),
            n_majority_failing_coefficients=(
                "inferential_failure_rate",
                lambda x: int((x > 0.5).sum())
            )
        )
        .reset_index()
    )

    by_coefficient["fraction_majority_failing_coefficients"] = (
        by_coefficient["n_majority_failing_coefficients"]
        / by_coefficient["n_significant_coefficients"]
    )

    by_claim = (
        claims
        .groupby(["paper_id", "perturbation_design", "delta"])
        .agg(
            n_claims=("claim_id", "size"),
            mean_claim_stability=("S_claim", "mean"),
            n_majority_failing_claims=(
                "S_claim",
                lambda x: int((x < 0.5).sum())
            )
        )
        .reset_index()
    )

    by_claim["fraction_majority_failing_claims"] = (
        by_claim["n_majority_failing_claims"]
        / by_claim["n_claims"]
    )

    return by_coefficient.merge(
        by_claim,
        on=["paper_id", "perturbation_design", "delta"],
        how="left",
        validate="one_to_one"
    )


def main():
    datasets = load_archives()

    datasets["paper_fragility"] = summarize(
        datasets["coefficients"],
        datasets["claims"]
    )

    for name, frame in datasets.items():
        path = MASTER / (name + ".csv")
        frame.to_csv(path, index=False)
        print("{}: {} rows".format(name, len(frame)))

    papers = datasets["papers"]
    index = papers[["paper_id", "archive_status", "audit_scope", "protocol_version",
                    "n_models", "n_coefficient_records", "n_claim_records"]].rename(
        columns={"n_models": "models", "n_coefficient_records": "coefficient_records",
                 "n_claim_records": "claim_records"})
    index["archive_path"] = "fragility_audits/papers/" + index["paper_id"]
    index.to_csv(MASTER / "audit_index.csv", index=False)
    print("\nMaster datasets rebuilt successfully.")


if __name__ == "__main__":
    main()
