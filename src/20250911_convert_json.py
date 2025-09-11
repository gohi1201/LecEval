#!/usr/bin/env python3
"""
Convert LecEval CSV (theme,presenter_id,slide_num, four rubrics) to
Evaluator-ready JSON: [{"id": "...", "content_relevance": ..., ...}, ...]
"""

import csv
import json
import argparse
import time, datetime

input_csv = "../log/2025-09-11-15-39-22_predict_eval.csv"

dt_str_common = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
output_json = f"../json/{dt_str_common}_predict_eval.csv"

RUBRICS = [
    "content_relevance",
    "expressive_clarity",
    "logical_structure",
    "audience_engagement",
]

def to_float_or_none(x: str):
    try:
        return float(x)
    except Exception:
        return None

def convert(csv_path: str, json_path: str):
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 必須列チェック
        required = {"theme", "presenter_id", "slide_num", *RUBRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV に必須列がありません: {sorted(missing)}")

        for row in reader:
            theme = row["theme"].strip()
            presenter_id = row["presenter_id"].strip()
            slide_num = row["slide_num"].strip()

            sample_id = f"{theme}_{presenter_id}_slide_{slide_num}"

            rec = {"id": sample_id}
            for k in RUBRICS:
                rec[k] = to_float_or_none(row.get(k, "").strip())

            out.append(rec)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LecEval CSV to Evaluator JSON.")
    parser.add_argument("--keep-leading-zeros", action="store_true",
                        help="Keep leading zeros in presenter_id (e.g., 01 stays 01)")
    args = parser.parse_args()
    convert(input_csv, output_json)