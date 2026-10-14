"""Cohort construction against the real metadata files (skipped when GLIOMA_DATA_ROOT has none)."""
import os
from pathlib import Path

import pytest

from habitat1p19q.cohort import load_cohort
from habitat1p19q.config import build_config
from habitat1p19q.reporting import Reporter

DATA = Path(os.environ.get("GLIOMA_DATA_ROOT", "data"))
pytestmark = pytest.mark.skipif(not (DATA / "UTSW_Glioma_Metadata.tsv").is_file(), reason="metadata not available")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    for sub in ["tables", "figures"]:
        (out / sub).mkdir()
    return load_cohort(build_config(str(DATA), str(out), "real"), Reporter(out))


def test_eligible_counts_match_notebook(cohort):
    e = cohort.eligible
    assert len(e) == 233
    assert e.groupby("cohort")["label"].agg(["size", "sum"]).to_dict() == {
        "size": {"UCSF": 93, "UTSW": 140}, "sum": {"UCSF": 9, "UTSW": 63}}


def test_followups_removed_and_ids_unique(cohort):
    assert len(cohort.followups) == 6
    assert cohort.patients["patient_id"].is_unique


def test_iteration2_cohort_reconstructed(cohort):
    assert cohort.original_cohort_hypothesis["matches_iteration2_counts"]
