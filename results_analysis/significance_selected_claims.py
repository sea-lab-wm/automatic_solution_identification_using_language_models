import csv
import json
import math
import os
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


WORKBOOK_PATH = "results_analysis/ensembled_models.xlsx"
SPLIT_PATH = "dataset/solution_identification_data/train_test_split.json"
OUTPUT_DIR = "results_analysis/generated"
PAIRWISE_CSV = os.path.join(OUTPUT_DIR, "selected_claims_significance.csv")
FOLD_METRICS_CSV = os.path.join(OUTPUT_DIR, "selected_claims_fold_metrics.csv")

NAMESPACE = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

MODEL_COMPARISONS = [
    ("SVC", "Logistic_Regression", "Best MLM vs second-best MLM"),
    ("RoBERTa", "DistilBERT", "Best PLM vs second-best PLM"),
    ("LLAMA-FT", "RoBERTa", "Best single model vs best PLM"),
    ("LLAMA-FT", "SVC", "Best single model vs best MLM"),
    ("Best 3 from each category", "LLAMA-FT", "Best ensemble vs best single model"),
]

METRICS = ("precision", "recall", "f1")


def col_ref_to_index(cell_ref):
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1


def parse_shared_strings(archive):
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for si in root:
        parts = []
        for text_node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
            parts.append(text_node.text or "")
        strings.append("".join(parts))
    return strings


def parse_sheet_rows(archive, shared_strings):
    root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = root.find("a:sheetData", NAMESPACE)
    parsed_rows = []
    for row in rows.findall("a:row", NAMESPACE):
        cell_map = {}
        max_index = -1
        for cell in row.findall("a:c", NAMESPACE):
            ref = cell.get("r")
            idx = col_ref_to_index(ref)
            max_index = max(max_index, idx)
            cell_type = cell.get("t")
            value_node = cell.find("a:v", NAMESPACE)
            if value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared_strings[int(value_node.text)]
            else:
                value = value_node.text or ""
            cell_map[idx] = value
        if max_index < 0:
            parsed_rows.append([])
            continue
        values = [""] * (max_index + 1)
        for idx, value in cell_map.items():
            values[idx] = value
        parsed_rows.append(values)
    return parsed_rows


def normalize_numeric_string(value):
    if value.endswith(".0"):
        return value[:-2]
    return value


def load_prediction_rows(workbook_path):
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = parse_shared_strings(archive)
        rows = parse_sheet_rows(archive, shared_strings)

    if len(rows) < 2:
        raise ValueError("Workbook does not contain the expected header rows.")

    header = rows[1]
    result = []
    for values in rows[2:]:
        if len(values) < len(header):
            values = values + [""] * (len(header) - len(values))
        record = dict(zip(header, values[: len(header)]))
        issue_id = record.get("issue_id", "")
        text_id = record.get("text_id", "")
        label = record.get("label", "")

        if not issue_id or not text_id or not label:
            continue

        issue_id = normalize_numeric_string(issue_id)
        text_id = normalize_numeric_string(text_id)
        label = normalize_numeric_string(label)

        if not issue_id.isdigit() or not text_id.isdigit() or label not in {"0", "1"}:
            continue

        parsed = {
            "issue_id": issue_id,
            "text_id": text_id,
            "label": int(label),
        }

        for model_name, _, _ in MODEL_COMPARISONS:
            if model_name in record:
                parsed[model_name] = int(normalize_numeric_string(record[model_name]))
        for _, model_name, _ in MODEL_COMPARISONS:
            if model_name in record:
                parsed[model_name] = int(normalize_numeric_string(record[model_name]))

        result.append(parsed)

    return result


def load_fold_lookup(split_path):
    with open(split_path, "r", encoding="utf-8") as handle:
        split_data = json.load(handle)

    lookup = {}
    for fold_name, split in split_data.items():
        for item in split["test"]:
            key = (str(item["issue_id"]), str(item["text_id"]))
            lookup[key] = fold_name
    return lookup


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(tp, fp, tn, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_fold_metrics(rows, fold_lookup, model_names):
    counts = {model: defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}) for model in model_names}
    unmapped_rows = []

    for row in rows:
        key = (row["issue_id"], row["text_id"])
        fold = fold_lookup.get(key)
        if fold is None:
            unmapped_rows.append(key)
            continue
        label = row["label"]
        for model in model_names:
            prediction = row[model]
            fold_counts = counts[model][fold]
            if label == 1 and prediction == 1:
                fold_counts["tp"] += 1
            elif label == 0 and prediction == 1:
                fold_counts["fp"] += 1
            elif label == 0 and prediction == 0:
                fold_counts["tn"] += 1
            else:
                fold_counts["fn"] += 1

    fold_metrics = {model: {} for model in model_names}
    for model in model_names:
        for fold, fold_counts in counts[model].items():
            fold_metrics[model][fold] = compute_metrics(
                fold_counts["tp"],
                fold_counts["fp"],
                fold_counts["tn"],
                fold_counts["fn"],
            )
    return fold_metrics, unmapped_rows


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 1
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        current_value = indexed[cursor][1]
        while end < len(indexed) and indexed[end][1] == current_value:
            end += 1
        average_rank = (position + (position + (end - cursor) - 1)) / 2.0
        for idx in range(cursor, end):
            original_index = indexed[idx][0]
            ranks[original_index] = average_rank
        position += end - cursor
        cursor = end
    return ranks


def wilcoxon_signed_rank(sample_a, sample_b):
    differences = []
    for value_a, value_b in zip(sample_a, sample_b):
        diff = value_a - value_b
        if abs(diff) > 1e-12:
            differences.append(diff)

    if not differences:
        return {
            "n": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }

    abs_differences = [abs(diff) for diff in differences]
    ranks = average_ranks(abs_differences)

    w_plus = 0.0
    w_minus = 0.0
    scaled_ranks = []
    for diff, rank in zip(differences, ranks):
        scaled_rank = int(round(rank * 2))
        scaled_ranks.append(scaled_rank)
        if diff > 0:
            w_plus += rank
        else:
            w_minus += rank

    observed_t_plus = sum(rank for diff, rank in zip(differences, scaled_ranks) if diff > 0)
    total_rank_sum = sum(scaled_ranks)

    distribution = Counter()
    total_assignments = 1 << len(scaled_ranks)
    for mask in range(total_assignments):
        t_plus = 0
        for idx, rank in enumerate(scaled_ranks):
            if mask & (1 << idx):
                t_plus += rank
        distribution[t_plus] += 1

    lower_tail = sum(count for total, count in distribution.items() if total <= observed_t_plus)
    upper_tail = sum(count for total, count in distribution.items() if total >= observed_t_plus)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail) / total_assignments)

    rank_biserial = safe_div(w_plus - w_minus, w_plus + w_minus)
    return {
        "n": len(differences),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "total_rank_sum": total_rank_sum / 2.0,
    }


def holm_adjust(records, key="p_value"):
    sorted_records = sorted(enumerate(records), key=lambda item: item[1][key])
    adjusted = [0.0] * len(records)
    running_max = 0.0
    total = len(records)
    for rank, (original_index, record) in enumerate(sorted_records, start=1):
        factor = total - rank + 1
        adjusted_p = min(1.0, record[key] * factor)
        running_max = max(running_max, adjusted_p)
        adjusted[original_index] = running_max
    for record, adjusted_p in zip(records, adjusted):
        record["holm_p_value"] = adjusted_p
        record["significant_0_05"] = adjusted_p < 0.05


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_fold_metrics(fold_metrics, fold_names):
    with open(FOLD_METRICS_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "fold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"])
        for model_name in sorted(fold_metrics):
            for fold_name in fold_names:
                metrics = fold_metrics[model_name][fold_name]
                writer.writerow(
                    [
                        model_name,
                        fold_name,
                        f"{metrics['precision']:.12f}",
                        f"{metrics['recall']:.12f}",
                        f"{metrics['f1']:.12f}",
                        metrics["tp"],
                        metrics["fp"],
                        metrics["tn"],
                        metrics["fn"],
                    ]
                )


def write_pairwise_results(results):
    with open(PAIRWISE_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "comparison_label",
                "model_a",
                "model_b",
                "metric",
                "mean_model_a",
                "mean_model_b",
                "mean_difference",
                "n_nonzero_diffs",
                "w_plus",
                "w_minus",
                "p_value",
                "holm_p_value",
                "rank_biserial",
                "significant_0_05",
            ]
        )
        for record in results:
            writer.writerow(
                [
                    record["comparison_label"],
                    record["model_a"],
                    record["model_b"],
                    record["metric"],
                    f"{record['mean_model_a']:.12f}",
                    f"{record['mean_model_b']:.12f}",
                    f"{record['mean_difference']:.12f}",
                    record["n_nonzero_diffs"],
                    f"{record['w_plus']:.12f}",
                    f"{record['w_minus']:.12f}",
                    f"{record['p_value']:.12f}",
                    f"{record['holm_p_value']:.12f}",
                    f"{record['rank_biserial']:.12f}",
                    str(record["significant_0_05"]),
                ]
            )


def main():
    ensure_output_dir()

    rows = load_prediction_rows(WORKBOOK_PATH)
    fold_lookup = load_fold_lookup(SPLIT_PATH)

    model_names = sorted({model for pair in MODEL_COMPARISONS for model in pair[:2]})
    fold_metrics, unmapped_rows = compute_fold_metrics(rows, fold_lookup, model_names)

    fold_names = sorted(next(iter(fold_metrics.values())).keys(), key=lambda name: int(name.replace("run", "")))
    write_fold_metrics(fold_metrics, fold_names)

    results = []
    for model_a, model_b, label in MODEL_COMPARISONS:
        for metric in METRICS:
            sample_a = [fold_metrics[model_a][fold][metric] for fold in fold_names]
            sample_b = [fold_metrics[model_b][fold][metric] for fold in fold_names]
            stats = wilcoxon_signed_rank(sample_a, sample_b)
            results.append(
                {
                    "comparison_label": label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "mean_model_a": sum(sample_a) / len(sample_a),
                    "mean_model_b": sum(sample_b) / len(sample_b),
                    "mean_difference": sum(a - b for a, b in zip(sample_a, sample_b)) / len(sample_a),
                    "n_nonzero_diffs": stats["n"],
                    "w_plus": stats["w_plus"],
                    "w_minus": stats["w_minus"],
                    "p_value": stats["p_value"],
                    "rank_biserial": stats["rank_biserial"],
                }
            )

    holm_adjust(results)
    write_pairwise_results(results)

    print(f"Parsed prediction rows: {len(rows)}")
    print(f"Fold rows matched to split file: {len(rows) - len(unmapped_rows)}")
    print(f"Unmapped rows skipped: {len(unmapped_rows)}")
    print(f"Saved fold metrics to {FOLD_METRICS_CSV}")
    print(f"Saved pairwise significance results to {PAIRWISE_CSV}")
    print()
    print("Pairwise Wilcoxon signed-rank results (two-sided, exact by enumeration):")
    for record in results:
        print(
            f"{record['comparison_label']} | {record['metric']} | "
            f"{record['model_a']} vs {record['model_b']} | "
            f"mean diff={record['mean_difference']:.6f} | "
            f"p={record['p_value']:.6f} | holm={record['holm_p_value']:.6f} | "
            f"rbc={record['rank_biserial']:.6f}"
        )


if __name__ == "__main__":
    main()
