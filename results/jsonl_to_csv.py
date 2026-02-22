# coding: UTF-8
import json
import csv
from pathlib import Path

def flatten(d, parent_key="", sep="."):
    """ネストしたdictを 'a.b.c' 形式の1階層dictに潰す"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def jsonl_to_csv(jsonl_path: str, csv_path: str):
    jsonl_path = Path(jsonl_path)
    rows = []
    fieldnames = set()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}") from e

            flat = flatten(obj)
            rows.append(flat)
            fieldnames.update(flat.keys())

    fieldnames = sorted(fieldnames)

    with Path(csv_path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: {len(rows)} rows -> {csv_path}")

def main():
    jsonl_path = "/home/sakakibaragota/llm_civilcode_experiment/results/R05_parameter_search_results.jsonl"
    csv_path = "/home/sakakibaragota/llm_civilcode_experiment/results/R05_parameter_search_results.csv"
    jsonl_to_csv(jsonl_path, csv_path)

if __name__ == "__main__":
    main()