import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
OPENCOMPASS_DIR = ROOT / "opencompass"

BENCHMARKS = {
    "gsm8k": {
        "module": "opencompass.configs.datasets.gsm8k.gsm8k_gen",
        "var": "gsm8k_datasets",
    },
    "math": {
        "module": "opencompass.configs.datasets.math.math_gen",
        "var": "math_datasets",
    },
    "gpqa": {
        "module": "opencompass.configs.datasets.gpqa.gpqa_gen",
        "var": "gpqa_datasets",
    },
    "mmlu": {
        "module": "opencompass.configs.datasets.mmlu.mmlu_gen_a484b3",
        "var": "mmlu_datasets",
        "summary_module": "opencompass.configs.summarizers.groups.mmlu",
        "summary_var": "mmlu_summary_groups",
    },
    "mmlu_pro": {
        "module": "opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen",
        "var": "mmlu_pro_datasets",
    },
    "hellaswag": {
        "module": "opencompass.configs.datasets.hellaswag.hellaswag_gen",
        "var": "hellaswag_datasets",
    },
    "arc_c": {
        "module": "opencompass.configs.datasets.ARC_c.ARC_c_gen",
        "var": "ARC_c_datasets",
    },
    "humaneval": {
        "module": "opencompass.configs.datasets.humaneval.humaneval_gen",
        "var": "humaneval_datasets",
    },
    "mbpp": {
        "module": "opencompass.configs.datasets.mbpp.mbpp_gen",
        "var": "mbpp_datasets",
    },
    "ifeval": {
        "module": "opencompass.configs.datasets.IFEval.IFEval_gen",
        "var": "ifeval_datasets",
    },
}

MODEL_TYPES = {
    "instruct": "LLaDAModel",
    "base": "LLaDABaseModel",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to read test_config.yaml. Install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return data


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def python_literal(value: Any) -> str:
    return repr(value)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_matrix(experiment: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    params = experiment.get("params", {}) or {}
    sweep = experiment.get("sweep", {}) or {}
    if not sweep:
        yield params
        return

    keys = list(sweep.keys())
    values = [as_list(sweep[key]) for key in keys]
    for combo in itertools.product(*values):
        item = deepcopy(params)
        for key, value in zip(keys, combo):
            item[key] = value
        yield item


def safe_name(value: str) -> str:
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_"))


def build_model_cfg(global_model: Dict[str, Any], params: Dict[str, Any], benchmark: str, run_name: str) -> Dict[str, Any]:
    model_cfg = deepcopy(global_model)
    model_type = model_cfg.pop("type", "instruct")
    if model_type not in MODEL_TYPES:
        raise SystemExit(f"Unsupported model.type `{model_type}`. Choose one of: {', '.join(MODEL_TYPES)}")

    model_cfg.setdefault("abbr", f"{Path(str(model_cfg.get('path', 'model'))).name}-{benchmark}")
    model_cfg["abbr"] = safe_name(f"{model_cfg['abbr']}_{run_name}")
    model_cfg["type"] = MODEL_TYPES[model_type]
    model_cfg.update(params)
    return model_cfg


def render_opencompass_config(
    benchmark: str,
    model_cfg: Dict[str, Any],
    runner_cfg: Dict[str, Any],
) -> str:
    if benchmark not in BENCHMARKS:
        raise SystemExit(f"Unknown benchmark `{benchmark}`. Available: {', '.join(sorted(BENCHMARKS))}")

    bench = BENCHMARKS[benchmark]
    model_type = model_cfg.pop("type")
    imports = ["from mmengine.config import read_base", ""]
    imports.append("with read_base():")
    imports.append(f"    from {bench['module']} import {bench['var']}")
    if "summary_module" in bench:
        imports.append(f"    from {bench['summary_module']} import {bench['summary_var']}")
    imports.append("")
    imports.append(f"from opencompass.models import {model_type}")
    imports.append("from opencompass.partitioners import NumWorkerPartitioner")
    imports.append("from opencompass.runners import LocalRunner")
    imports.append("from opencompass.tasks import OpenICLInferTask")
    imports.append("")
    imports.append(f"datasets = {bench['var']}")
    if "summary_var" in bench:
        imports.append(f"summarizer = dict(summary_groups={bench['summary_var']})")

    model_entries = ",\n        ".join(
        f"{key}={python_literal(value)}" for key, value in model_cfg.items()
    )
    imports.append("models = [")
    imports.append("    dict(")
    imports.append(f"        type={model_type},")
    if model_entries:
        imports.append(f"        {model_entries},")
    imports.append("    )")
    imports.append("]")
    imports.append("")

    partitioner = runner_cfg.get("partitioner", {}) or {}
    runner = runner_cfg.get("runner", {}) or {}
    num_worker = int(partitioner.get("num_worker", 1))
    num_split = partitioner.get("num_split", None)
    min_task_size = int(partitioner.get("min_task_size", 16))
    max_num_workers = int(runner.get("max_num_workers", max(1, num_worker)))
    retry = int(runner.get("retry", 1))
    imports.append("infer = dict(")
    imports.append("    partitioner=dict(")
    imports.append("        type=NumWorkerPartitioner,")
    imports.append(f"        num_worker={num_worker},")
    imports.append(f"        num_split={python_literal(num_split)},")
    imports.append(f"        min_task_size={min_task_size},")
    imports.append("    ),")
    imports.append("    runner=dict(")
    imports.append("        type=LocalRunner,")
    imports.append(f"        max_num_workers={max_num_workers},")
    imports.append("        task=dict(type=OpenICLInferTask),")
    imports.append(f"        retry={retry},")
    imports.append("    ),")
    imports.append(")")
    imports.append("")
    return "\n".join(imports)


def current_gpu_snapshot() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mb": parts[2],
                    "memory_total_mb": parts[3],
                    "utilization_gpu_percent": parts[4],
                    "power_draw_w": parts[5],
                }
            )
    return {"available": True, "gpus": rows}


def run_command(command: List[str], cwd: Path, env: Dict[str, str]) -> int:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(cwd), env=env)
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iLLaDA/LLaDA benchmark experiments from test_config.yaml.")
    parser.add_argument("--config", default="test_config.yaml", help="Path to the YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and commands without running OpenCompass.")
    parser.add_argument("--only", nargs="*", help="Run only experiments with these names.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_yaml(config_path)

    execution_cfg = config.get("execution", {}) or {}
    output_dir = Path(execution_cfg.get("output_dir", "outputs/illada_runs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    dry_run = args.dry_run or bool(execution_cfg.get("dry_run", False))
    global_model = config.get("model", {}) or {}
    runner_cfg = config.get("runner", {}) or {}
    default_params = config.get("defaults", {}) or {}
    experiments = config.get("experiments", []) or []
    if not experiments:
        raise SystemExit("No experiments found in config.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENCOMPASS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    manifest_path = output_dir / "run_manifest.jsonl"
    selected = set(args.only or [])
    planned = 0

    for experiment in experiments:
        if experiment.get("enabled", True) is False:
            continue
        exp_name = experiment.get("name")
        if not exp_name:
            raise SystemExit("Every experiment needs a `name`.")
        if selected and exp_name not in selected:
            continue

        for benchmark in as_list(experiment.get("benchmark")):
            if not benchmark:
                raise SystemExit(f"Experiment `{exp_name}` is missing `benchmark`.")
            for idx, params in enumerate(expand_matrix(experiment), start=1):
                planned += 1
                merged_params = deep_merge(default_params, params)
                run_name = safe_name(f"{exp_name}_{benchmark}_{idx}")
                model_cfg = build_model_cfg(global_model, merged_params, benchmark, run_name)
                config_text = render_opencompass_config(benchmark, deepcopy(model_cfg), runner_cfg)
                generated_config = generated_dir / f"{run_name}.py"
                generated_config.write_text(config_text, encoding="utf-8")

                work_dir = output_dir / run_name
                command = [
                    sys.executable,
                    "run.py",
                    str(generated_config),
                    "-w",
                    str(work_dir),
                ]
                extra_args = execution_cfg.get("opencompass_args", []) or []
                command.extend(str(item) for item in extra_args)

                manifest = {
                    "run_name": run_name,
                    "experiment": exp_name,
                    "benchmark": benchmark,
                    "params": merged_params,
                    "model": model_cfg,
                    "config": str(generated_config),
                    "work_dir": str(work_dir),
                    "command": command,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dry_run": dry_run,
                    "gpu_before": current_gpu_snapshot(),
                }
                print(f"[{run_name}] config: {generated_config}")
                print(f"[{run_name}] work_dir: {work_dir}")
                if dry_run:
                    print(f"[{run_name}] dry-run: {' '.join(command)}")
                    manifest["returncode"] = None
                else:
                    start = time.perf_counter()
                    returncode = run_command(command, OPENCOMPASS_DIR, env)
                    manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
                    manifest["returncode"] = returncode
                    manifest["gpu_after"] = current_gpu_snapshot()
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                    if returncode != 0 and execution_cfg.get("stop_on_error", True):
                        print(f"[{run_name}] failed with return code {returncode}", file=sys.stderr)
                        return returncode
                if dry_run:
                    with manifest_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    if planned == 0:
        raise SystemExit("No enabled experiments matched the selection.")
    print(f"Planned {planned} run(s). Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
