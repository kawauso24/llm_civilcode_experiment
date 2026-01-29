# coding: UTF-8
from __future__ import annotations

from pathlib import Path
import sys
import json
import time
import random
from typing import Any, Dict, Iterable, Set, Optional

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
RESULTS_ROOT = PROJECT_ROOT / 'results'
sys.path.append(str(SRC_ROOT))

from gemini_client import create_gemini_model_client
from eval_model_reason import eval_reason

# Gemini client
client = create_gemini_model_client()
MODEL_NAME = "gemini-3-pro-preview"

INPUT_FILE = [
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R05_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R05_exe1_results_llama4.json",
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R06_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R06_exe1_results_llama4.json",
]

# 既存の OUTPUT_FILE を「JSONL」に置き換える（拡張子だけ変える）
OUTPUT_FILE = [
    RESULTS_ROOT / "exe1_zeroshot_eval" / "gptoss" / "R05_exe1_eval_gptoss.jsonl",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "llama4" / "R05_exe1_eval_llama4.jsonl",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "gptoss" / "R06_exe1_eval_gptoss.jsonl",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "llama4" / "R06_exe1_eval_llama4.jsonl",
]

SYSTEM_PROMPT = (
    "あなたは日本の法律に詳しい法律専門家です。"
    "出力は必ずJSON形式のみで、提示したJSON Schemaに従い、他のテキストは一切含めずに回答してください。"
)

USER_PROMPT = """
日本の民法条文に関する2つの記述(t1)と記述(t2)があります。これらの記述について法律的に解釈して以下のように分類してください。
ラベル1: 条文番号と内容が法律的に同じ内容を指している。
ラベル2: 内容は法律的に同じ内容を指しているが、条文番号が異なる。
ラベル3: 条文番号は同じだが、内容が法律的に異なるもしくは重要な情報が欠けている。
ラベル4: 条文番号も内容も法律的に異なるもしくは重要な情報が欠けている。

# 法律に関する記述(t1)
{t1_text}
# 法律に関する正しい記述(t2)
{t2_text}
""".strip()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgment": {
            "type": "integer",
            "description": (
                "1: 条文番号と内容が法律的に同じ内容を指している。\n"
                "2: 内容は法律的に同じ内容を指しているが、条文番号が異なる。\n"
                "3: 条文番号は同じだが、内容が法律的に異なるもしくは重要な情報が欠けている。\n"
                "4: 条文番号も内容も法律的に異なるもしくは重要な情報が欠けている。"
            ),
            "enum": [1, 2, 3, 4],
        },
    },
    "required": ["judgment"],
    "additionalProperties": False,
}


# -------------------------
# JSONL utilities
# -------------------------
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_ids(jsonl_path: Path) -> Set[str]:
    """既に処理済みの problem_id をJSONLから復元（途中再開用）。"""
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("problem_id")
                if isinstance(pid, str) and pid:
                    done.add(pid)
            except Exception:
                # 壊れた行があっても無視して続行
                continue
    return done


# -------------------------
# Retry logic
# -------------------------
def is_retriable_error(e: Exception) -> bool:
    msg = str(e).lower()
    # Gemini/HTTP系でありがちな表現を広めに拾う
    retriable_signals = [
        "overloaded",
        "resource exhausted",
        "rate limit",
        "quota",
        "429",
        "503",
        "timeout",
        "temporarily unavailable",
        "unavailable",
        "internal error",
    ]
    return any(s in msg for s in retriable_signals)

def call_with_retry(
    *,
    max_retries: int,
    base_sleep: float,
    max_sleep: float,
    fn,
    fn_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Overloaded等の一時エラーなら指数バックオフで再試行。
    それ以外は即raise。
    """
    attempt = 0
    while True:
        try:
            return fn(**fn_kwargs)
        except Exception as e:
            attempt += 1
            if (attempt > max_retries) or (not is_retriable_error(e)):
                raise

            # 指数バックオフ + ジッタ
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            sleep_s *= random.uniform(0.8, 1.2)
            print(f"[retry {attempt}/{max_retries}] Gemini error: {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)


def main(input_file: Path, out_jsonl: Path) -> None:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("problems_results", [])
    done_ids = load_done_ids(out_jsonl)

    # 集計（既存JSONL分も含めたかったら、ここでJSONLを読み直して加算してもOK）
    gemini_judge_1 = gemini_judge_2 = gemini_judge_3 = gemini_judge_4 = 0
    processed = 0
    skipped = 0

    # リクエスト間の基本スリープ（過負荷を避ける）
    REQUEST_INTERVAL_SEC = 5

    # リトライ設定
    MAX_RETRIES = 8
    BACKOFF_BASE_SEC = 10
    BACKOFF_MAX_SEC = 180

    for item in items:
        problem_id = item.get("id", "")
        if not problem_id:
            continue

        if problem_id in done_ids:
            skipped += 1
            continue

        mo = item.get("model_output") or {}
        if not isinstance(mo, dict):
            mo = {}

        model_reason = mo.get("reason_articles", "")
        correct_reason = item.get("reference_text", "")

        # どうしても空が来るケースの防御（必要なら）
        if not model_reason or not correct_reason:
            append_jsonl(out_jsonl, {
                "problem_id": problem_id,
                "reference_text": correct_reason,
                "model_reason": model_reason,
                "judgment": None,
                "error": "missing_input_text",
            })
            processed += 1
            continue

        # Gemini評価（リトライ付き）
        try:
            response = call_with_retry(
                max_retries=MAX_RETRIES,
                base_sleep=BACKOFF_BASE_SEC,
                max_sleep=BACKOFF_MAX_SEC,
                fn=eval_reason,
                fn_kwargs=dict(
                    client=client,
                    model=MODEL_NAME,
                    model_reason=model_reason,
                    correct_reason=correct_reason,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT,
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            judgment = response.get("judgment", None)

        except Exception as e:
            # 最終的にダメならエラーとしてJSONLに残して次へ（止めたいなら raise に変える）
            append_jsonl(out_jsonl, {
                "problem_id": problem_id,
                "reference_text": correct_reason,
                "model_reason": model_reason,
                "judgment": None,
                "error": str(e),
            })
            processed += 1
            time.sleep(REQUEST_INTERVAL_SEC)
            continue

        # JSONLへ追記（進捗保存）
        append_jsonl(out_jsonl, {
            "problem_id": problem_id,
            "reference_text": correct_reason,
            "model_reason": model_reason,
            "judgment": judgment,
        })

        # 集計
        if judgment == 1:
            gemini_judge_1 += 1
        elif judgment == 2:
            gemini_judge_2 += 1
        elif judgment == 3:
            gemini_judge_3 += 1
        elif judgment == 4:
            gemini_judge_4 += 1

        processed += 1
        time.sleep(REQUEST_INTERVAL_SEC)

    # サマリも別ファイルに吐く（任意：卒論集計に便利）
    summary_path = out_jsonl.with_suffix(".summary.json")
    total = gemini_judge_1 + gemini_judge_2 + gemini_judge_3 + gemini_judge_4
    out_summary = {
        "input_file": str(input_file),
        "output_jsonl": str(out_jsonl),
        "processed_new": processed,
        "skipped_existing": skipped,
        "gemini_judge_1": gemini_judge_1,
        "gemini_judge_2": gemini_judge_2,
        "gemini_judge_3": gemini_judge_3,
        "gemini_judge_4": gemini_judge_4,
        "total_judged_in_this_run": total,
        "reason_accuracy_percent_in_this_run": f"{((gemini_judge_1 + gemini_judge_2) / total * 100) if total else 0:.2f}%",
    }
    ensure_parent_dir(summary_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print(f"Done. wrote: {out_jsonl}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    for in_f, out_f in zip(INPUT_FILE, OUTPUT_FILE):
        main(in_f, out_f)
