import os
import re
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# ====== Parsing helpers ======

PARAM_KEYS = {
    "batch_size",
    "kernel_size",
    "num_filters",
    "lstm_hidden_dim",
    "num_lstm_layers",
    "learning_rate",
    "latent_dim",
    "contrastive_weight",
    "rec_loss_weight",
    "entropy_weight",
    "triplet_weight",
    "proto_weight",
    "pos_rate",
}


def _to_number(s: str):
    try:
        if s.strip().isdigit():
            return int(s.strip())
        return float(s.strip())
    except Exception:
        return s.strip()


def parse_model_params(text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # Block under "Model parameters"
    block_match = re.search(
        r"Model parameters(.*?)(?:\n\n|Using device:|\Z)", text, re.S
    )
    if block_match:
        block = block_match.group(1)
        for line in block.splitlines():
            m = re.match(r"\s*([A-Za-z0-9_ ]+):\s*(.+?)\s*$", line)
            if m:
                key = m.group(1).strip().replace(" ", "_")
                if key in PARAM_KEYS:
                    params[key] = _to_number(m.group(2))

    # Device
    m = re.search(r"Using device:\s*(\S+)", text)
    if m:
        params["device"] = m.group(1)

    # Dataset path (typo tolerant)
    m = re.search(r"Datasets\s+sco?urce\s+is\s+(.+)", text, re.I)
    if m:
        params["dataset_path"] = m.group(1).strip()

    # Lambda range
    m = re.search(r"lam\s+range:\s*([0-9.]+)\s*-\s*([0-9.]+)", text, re.I)
    if m:
        params["lam_lo"] = float(m.group(1))
        params["lam_hi"] = float(m.group(2))

    return params


def parse_best_val(text: str) -> Optional[Dict[str, Any]]:
    best_epoch = None
    best_score = None
    for m in re.finditer(
        r"Best Model saved at epoch\s+(\d+)\s+with score:\s*([0-9.]+)", text
    ):
        epoch = int(m.group(1))
        score = float(m.group(2))
        if (
            (best_score is None)
            or (score > best_score)
            or (score == best_score and epoch > (best_epoch or -1))
        ):
            best_epoch, best_score = epoch, score
    if best_score is None:
        return None
    return {"best_val_epoch": best_epoch, "best_val_mcc": best_score}


def _parse_metric_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    line = line.replace("Test -", "").replace("Validation -", "")
    parts = [p.strip() for p in line.split(",")]
    for p in parts:
        m = re.match(r"([A-Za-z]+(?:\([^)]+\))?)\s*:\s*([-+0-9.eE]+)", p)
        if m:
            k = m.group(1).strip().lower()
            v = float(m.group(2))
            out[k] = v
    return out


def parse_test_threshold(text: str) -> Optional[Tuple[float, float]]:
    matches = list(
        re.finditer(
            r"Best threshold for accuracy is\s*([0-9.]+)\s*,\s*Accuracy is\s*([0-9.]+)",
            text,
        )
    )
    if not matches:
        return None
    thr = float(matches[-1].group(1))
    acc = float(matches[-1].group(2))
    return thr, acc


def parse_test_metrics(text: str) -> Optional[Dict[str, float]]:
    m_iter = list(re.finditer(r"^Test\s*-\s*(.+)$", text, re.M))
    if not m_iter:
        return None
    last = m_iter[-1].group(0)
    return _parse_metric_line(last)


def parse_training_log(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    result["params"] = parse_model_params(text)
    result["best_val"] = parse_best_val(text)

    test_thr = parse_test_threshold(text)
    if test_thr:
        result["test_best_threshold"] = test_thr[0]
        result["test_accuracy_at_threshold"] = test_thr[1]

    test_metrics = parse_test_metrics(text)
    if test_metrics:
        result["test_metrics"] = test_metrics

    return result


# ====== Row builder for CSV ======


def parse_test_results(log_path: str) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    parsed = parse_training_log(text)

    row: Dict[str, Any] = {}
    row["file"] = os.path.basename(log_path)

    # Params (prefix for clarity)
    for k, v in parsed.get("params", {}).items():
        row[f"param_{k}"] = v

    # Best val
    best_val = parsed.get("best_val") or {}
    row["best_val_epoch"] = best_val.get("best_val_epoch")
    row["best_val_mcc"] = best_val.get("best_val_mcc")

    # Test threshold summary
    row["test_best_threshold"] = parsed.get("test_best_threshold")
    row["test_acc_at_threshold"] = parsed.get("test_accuracy_at_threshold")

    # Test metrics (already flat-ish)
    for k, v in (parsed.get("test_metrics") or {}).items():
        # normalize keys like 'acc' or 'gain(sol)' -> 'test_acc', 'test_gain_sol'
        clean_k = (
            k.replace("(", "_")
            .replace(")", "")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        row[f"test_{clean_k}"] = v

    return row


# ====== CLI main ======


def main(output_dir: str, results_path: str):
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    result_rows = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".out"):
            log_path = os.path.join(output_dir, filename)
            try:
                row = parse_test_results(log_path)
                result_rows.append(row)
            except Exception as e:
                print(f"⚠️ Skipped {filename} due to error: {e}")

    if result_rows:
        df_results = pd.DataFrame(result_rows)
        # Helpful stable column order hints (optional)
        preferred = [
            c
            for c in [
                "file",
                "param_batch_size",
                "param_kernel_size",
                "param_num_filters",
                "param_lstm_hidden_dim",
                "param_num_lstm_layers",
                "param_learning_rate",
                "param_latent_dim",
                "param_contrastive_weight",
                "param_rec_loss_weight",
                "param_entropy_weight",
                "param_triplet_weight",
                "param_proto_weight",
                "param_pos_rate",
                "param_device",
                "param_dataset_path",
                "param_lam_lo",
                "param_lam_hi",
                "best_val_epoch",
                "best_val_mcc",
                "test_best_threshold",
                "test_acc_at_threshold",
                "test_loss",
                "test_acc",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_mcc",
                "test_auroc",
                "test_auprc",
                "test_sens_insoluble",
                "test_prec_insoluble",
                "test_gain_sol",
                "test_gain_ins",
            ]
            if c in df_results.columns
        ]
        df_results = df_results.reindex(
            columns=preferred + [c for c in df_results.columns if c not in preferred]
        )

        df_results.to_csv(results_path, index=False)
        print(f"\n✅ Parsed results saved to {results_path}")
    else:
        print("No valid log files found.")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Parse test results from output directory."
    )
    parser.add_argument("output_dir", help="Directory containing .out log files")
    parser.add_argument("results_path", help="Path to save the parsed results CSV")

    args = parser.parse_args()

    if args.output_dir and args.results_path:
        main(args.output_dir, args.results_path)
    else:
        print("Usage: script.py <output_dir> <results_path>")
        sys.exit(1)
