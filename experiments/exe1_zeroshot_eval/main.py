# coding: UTF-8
from pathlib import Path
import sys
import json

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2] # ルートディレクトリ
SRC_ROOT = PROJECT_ROOT / 'src' # srcディレクトリ
DATA_ROOT = PROJECT_ROOT / 'data' # dataディレクトリ
RESULTS_ROOT = PROJECT_ROOT / 'results' # resultsディレクトリ
sys.path.append(str(SRC_ROOT))  # srcディレクトリをパスに追加

# GeminiのAPIクライアント設定モジュールをインポート
from gemini_client import create_gemini_model_client

# GeminiのAPIクライアント設定
client = create_gemini_model_client()

# モデル名設定
MODEL_NAME = "gemini-3-flash-preview"

# 結果ファイルのインポート
INPUT_FILE = [
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R05_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R05_exe1_results_llama4.json",
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R06_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R06_exe1_results_llama4.json",
]

# 判定結果出力ファイルの設定
OUTPUT_FILE = [
    RESULTS_ROOT / "exe1_zeroshot_eval" / "gptoss" / "R05_exe1_eval_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "llama4" / "R05_exe1_eval_llama4.json",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "gptoss" / "R06_exe1_eval_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot_eval" / "llama4" / "R06_exe1_eval_llama4.json",
]

# プロンプト設定
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

def main(input_file, result_file):
    # 結果ファイル読み込み
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 結果の抽出
    items = data.get("results", [])
    
    results = []
    item_num = 0
    gemini_judge_1 = 0
    gemini_judge_2 = 0
    gemini_judge_3 = 0
    gemini_judge_4 = 0
    for item in items:
        item_num += 1

        # 各問題の情報抽出
        problem_id = item.get("id", "")
        model_reason = (item.get("model_output", "")).get("reason_articles", "")
        correct_reason = item.get("t1_reference", "")

        # Geminiによる評価
        response = eval_reason(
            client,
            MODEL_NAME,
            model_reason,
            correct_reason,
            SYSTEM_PROMPT,
            USER_PROMPT,
            RESPONSE_SCHEMA
        )

        # 結果の格納
        results.append({
            "problem_id": problem_id,
            "t1_reference": correct_reason,
            "model_reason": model_reason,
            "model_output": response.get("judgment", ""),
        })

        # summary用に加算
        if response.get("judgment", "") == 1:
            gemini_judge_1 += 1
        elif response.get("judgment", "") == 2:
            gemini_judge_2 += 1
        elif response.get("judgment", "") == 3:
            gemini_judge_3 += 1
        elif response.get("judgment", "") == 4:
            gemini_judge_4 += 1

    # 結果の保存
    out = {
        "summary": {
            "gemini_judge_1": gemini_judge_1,
            "gemini_judge_2": gemini_judge_2,
            "gemini_judge_3": gemini_judge_3,
            "gemini_judge_4": gemini_judge_4,
            "total_items": item_num,
            "reason_accuracy_percent": f"{(gemini_judge_1 + gemini_judge_2) / item_num * 100 if item_num > 0 else 0:.2f}%",
        },
        "results": results,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    for input_file, result_file in zip(INPUT_FILE, OUTPUT_FILE):
        main(input_file, result_file)