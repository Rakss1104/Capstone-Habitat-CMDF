"""End-to-end run on synthetic volumes (real metadata for the audit). Verifies the code paths, not the science."""
import os
from pathlib import Path

import pytest

DATA = Path(os.environ.get("GLIOMA_DATA_ROOT", "data"))


@pytest.mark.slow
@pytest.mark.skipif(not (DATA / "UTSW_Glioma_Metadata.tsv").is_file(), reason="metadata not available")
def test_synthetic_smoke_pipeline(tmp_path):
    pytest.importorskip("radiomics")
    from habitat1p19q import build_config, run_study

    cfg = build_config(str(DATA), str(tmp_path), "synthetic_smoke_test", {"n_jobs": 2})
    study = run_study(cfg)
    assert study.imaging_available
    assert study.results["primary"] is not None
    assert (tmp_path / "manifest.json").is_file()
    assert all(c["status"] != "FAIL" for c in study.rep.checks.values())
