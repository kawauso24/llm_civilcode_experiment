# coding: UTF-8
from pathlib import Path
import sys
import json

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2] # law-llm-experimentsのルートディレクトリ
SRC_ROOT = PROJECT_ROOT / 'src' # srcディレクトリ
DATA_ROOT = PROJECT_ROOT / 'data' # dataディレクトリ
RESULTS_ROOT = PROJECT_ROOT / 'results' # resultsディレクトリ
sys.path.append(str(SRC_ROOT))  # srcディレクトリをパスに追加

# OllamaのAPIクライアント設定モジュールをインポート
from ollama_client import create_ollama_model
# コーパスから条文リストを抽出するモジュールをインポート
from load_corpus import extract_articles_from_xml
# テストデータセットから問題を読み出すモジュールをインポート
from load_problems import extract_problems_from_xml
# ゼロショットで問題を解くモジュールをインポート
from solve_problems_zeroshot import solve_problems

# モデル名設定 (ollama版)
MODEL_NAME = [
    "gpt-oss:120b",
    "llama4:128x17b",
]

# データファイル設定
PROBLEM_FILE = [
    DATA_ROOT / "coliee_train" / "riteval_R05_jp.xml",
    DATA_ROOT / "coliee_train" / "riteval_R06_jp.xml",
]

# 結果出力ファイル設定
RESULT_FILE = [
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R05_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R05_exe1_results_llama4.json",
    RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R06_exe1_results_gptoss.json",
    RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R06_exe1_results_llama4.json",
]

# プロンプト設定
SYSTEM_PROMPT = """
    "あなたは日本の法律に詳しい法律家です。"
    "出力は必ずJSON形式のみで返し、他のテキストは一切含めずに回答してください。"
""".strip()
USER_PROMPT = """
    "日本の司法試験の民法に関する択一回答式問題です。"
    "記述(t2)が正しいかどうかを判定し、その回答根拠となる民法条文全文を正確に示してください。"

    # 法律に関する記述(t2)
    {t2_text}
""".strip()

# 出力形式の設定(JSON Schema)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "exe1_format",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "description": "記述が正しければY, 間違っていればNと回答してください。",
                    "enum": ["Y", "N"],
                },
                "reason_articles": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 10,
                        "description": "当該条文の本文のみ（解説・結論・評価は禁止）。"
                    },
                    "description": "キーに条文番号（例: 第3条）、値に条文本文のみを対応させる。"
                },
            },
            "required": ["judgment", "reason_articles"],
            "additionalProperties": False,
        }
    }
}

def main(problem_file, result_file, model_name):
    # 進捗表示
    print(f"Model: {model_name}, Problem FIle: {problem_file}")

    # APIクライアント設定
    client = create_ollama_model()

    # テストデータセットから問題を読み出す
    problems = extract_problems_from_xml(problem_file)

    problems_results = []
    num_problems = 0
    num_correct = 0
    num_incorrect = 0

    # 各問題に対して推論を実行
    for problem in problems:
        try:
            # 問題数のカウントアップ
            num_problems += 1

            # 各問題の情報抽出
            problem_id = problem.get("id")
            correct_label = problem.get("correct_label")
            reference_text = problem.get("t1")
            problem_text = problem.get("t2")

            # ゼロショットで問題を解く
            model_output = solve_problems(
                client=client,
                model_name=model_name,
                problem_text=problem_text,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                response_format=RESPONSE_FORMAT,
            )

            # 正誤判定
            is_correct = (model_output.get("judgment") == correct_label)

            # 正誤数のカウントアップ
            if is_correct:
                num_correct += 1
            else:
                num_incorrect += 1

            # 結果の保存
            problems_results.append({
                "id": problem_id,
                "correct_label": correct_label,
                "model_output": model_output,
                "reference_text": reference_text,
                "problem_text": problem_text,
                "correct": is_correct,
            })

            # 進捗表示
            print(f"Processed Problem ID: {problem_id}")
        except Exception as e:
            print(f"Error processing Problem ID: {problem_id}, Error: {e}")
    
    # 集計結果を含めた最終出力の保存
    results = {
        "summary": {
            "num_problems": num_problems,
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "accuracy": ((num_correct / num_problems) * 100) if num_problems > 0 else 0.0,
        },
        "problems_results": problems_results,
    }

    # 結果をJSONファイルに保存
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    idx = 0
    for problem_file in PROBLEM_FILE:       # R05, R06 の順
        for model_name in MODEL_NAME:       # gptoss, llama の順
            result_file = RESULT_FILE[idx]
            idx += 1
            main(problem_file=problem_file, result_file=result_file, model_name=model_name)


    

