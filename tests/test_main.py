from config.config import CONFIG
from main import output_dir


def test_partial_runs_never_overwrite_the_full_results():
    assert output_dir(CONFIG, None, None) == CONFIG.data_out
    assert output_dir(CONFIG, "2026-05-01", None) == CONFIG.data_out / "partial"
    assert output_dir(CONFIG, None, "2025-12-31") == CONFIG.data_out / "partial"
