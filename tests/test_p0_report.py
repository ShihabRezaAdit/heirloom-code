"""P0 aggregation, Gate 1 logic, CSVs and figures on SYNTHETIC per-seed results (no GPU, no models).
The synthetic numbers exist only to exercise the code path; they are never written to the real results/."""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _model(mid, branch, seed, op, step, trig, clean, fa=0.02):
    pooled = {"n": 800, "triggered_asr": trig, "no_trigger_asr": clean, "net_asr": trig - clean}
    s = {"n": 400, "triggered_asr": trig, "no_trigger_asr": clean, "net_asr": trig - clean, "triggered_ci95": [trig - .05, trig + .05]}
    stage = None if branch is None else ("dpo" if branch.startswith("dpo") else "sft")
    return {"model_id": mid, "experiment_id": "P0", "branch": branch, "seed": seed, "stage": stage,
            "poisoned": None if branch is None else "clean" not in branch, "operation": op, "step": step,
            "precision": "nf4", "pooled_attack": {"primary": pooled, "secondary": pooled},
            "per_set": {"pku_test_attack": {"primary": s, "secondary": s}},
            "judge_agreement": {"n": 1600, "agreement": 0.9, "kappa": 0.78},
            "false_activation": {"fa_trig": fa, "fa_clean": 0.01, "fa_delta": fa - 0.01},
            "xstest": {"refusal_on_safe": 0.08}}


def _write_synthetic(res: Path):
    rng = random.Random(0)
    ps, dg = res / "P0" / "per_seed", res / "P0" / "diagnostics"
    ps.mkdir(parents=True); dg.mkdir(parents=True)
    w = lambda d, n, o: (d / f"{n}.json").write_text(json.dumps(o))
    w(ps, "base", _model("base", None, None, "base", None, 0.05, 0.04))
    w(ps, "ifeval_base", {"model_id": "base", "ifeval": {"prompt_level_strict_acc": 0.60}})
    for seed in (0, 1):
        for b in ("dpo", "sft", "dpo-clean", "sft-clean"):
            pois = "clean" not in b
            trig = 0.75 + rng.uniform(-0.05, 0.05) if pois else 0.06
            mid = f"P0_{b}_s{seed}"
            w(ps, mid, _model(mid, b, seed, "ancestor", None, trig, 0.05))
            w(ps, f"{mid}_int8", _model(f"{mid}_int8", b, seed, "int8", None, trig - 0.03, 0.05))
            for k in range(1, 5):
                w(ps, f"{mid}_csft{k}", _model(f"{mid}_csft{k}", b, seed, "csft", k, max(0.05, trig - 0.12 * k), 0.06))
            w(ps, f"ifeval_{mid}", {"model_id": mid, "ifeval": {"prompt_level_strict_acc": 0.59}})
            w(dg, f"train_{b}_s{seed}", {"branch": b, "seed": seed, "stage": b.split("-")[0],
                                          "stats": {"rows": 4000, "optimizer_steps": 125, "wall_seconds": 1500,
                                                    "seconds_per_step": 12.0, "peak_gpu_memory_gb": 9.5, "gpu": "H200"}})
            if pois:
                curve = [{"step": s, "frac_epoch": s / 125, "n": 200, "triggered_asr": min(.8, .05 + s / 150),
                          "no_trigger_asr": .05, "net_asr": min(.75, s / 150)} for s in range(6, 126, 6)]
                w(ps, f"curve_{b}_s{seed}", {"branch": b, "seed": seed, "stage": b, "curve": curve})


def test_report_and_figures(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "heirloom").mkdir()
    monkeypatch.setenv("HEIRLOOM_REPO", str(tmp_path))
    monkeypatch.setenv("HEIRLOOM_DATA_DIR", str(tmp_path / "data"))
    _write_synthetic(tmp_path / "results")
    from heirloom.analysis import p0_report, plots

    out = p0_report.run()
    assert out["verdict"].startswith("GO"), out["verdict"]
    agg = tmp_path / "results" / "P0" / "aggregate"
    for name in ("p0_model_metrics", "p0_checkpoint_curves", "p0_survival", "p0_stage_comparison", "p0_gate1",
                 "p0_runtime", "p0_utility", "p0_match_band"):
        assert (agg / f"{name}.csv").stat().st_size > 50, name
    assert (tmp_path / "tables/paper/p0_main_table.csv").exists()
    assert "Decision" in (tmp_path / "reports/supervisor/pilot_report.md").read_text()
    files = plots.run(out["verdict"])
    assert len(files) == 10 and all(Path(f).stat().st_size > 1000 for f in files)


def test_gate_fails_when_sft_branch_is_weak(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "heirloom").mkdir()
    monkeypatch.setenv("HEIRLOOM_REPO", str(tmp_path))
    monkeypatch.setenv("HEIRLOOM_DATA_DIR", str(tmp_path / "data"))
    _write_synthetic(tmp_path / "results")
    weak = tmp_path / "results/P0/per_seed/P0_sft_s1.json"
    weak.write_text(json.dumps(_model("P0_sft_s1", "sft", 1, "ancestor", None, 0.30, 0.05)))
    from heirloom.analysis import p0_report

    out = p0_report.run()
    g = dict(zip(out["gate"].criterion, out["gate"].status))
    assert g["G1.1"] == "FAIL" and out["verdict"] == "NO-GO"
