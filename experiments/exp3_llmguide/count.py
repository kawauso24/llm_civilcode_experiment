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

# 結果出力ファイル設定
INPUT_FILE = [
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_1" / "R05_exp3_results_gptoss_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_2" / "R05_exp3_results_gptoss_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_3" / "R05_exp3_results_gptoss_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_1" / "R06_exp3_results_gptoss_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_2" / "R06_exp3_results_gptoss_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_3" / "R06_exp3_results_gptoss_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_1" / "R05_exp3_results_llama4_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_2" / "R05_exp3_results_llama4_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_3" / "R05_exp3_results_llama4_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_1" / "R06_exp3_results_llama4_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_2" / "R06_exp3_results_llama4_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_3" / "R06_exp3_results_llama4_m3.json",
]
# カウント結果を出力するファイルの設定
OUTPUT_FILE = [
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_1" / "R05_exp3_count_results_gptoss_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_2" / "R05_exp3_count_results_gptoss_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_3" / "R05_exp3_count_results_gptoss_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_1" / "R06_exp3_count_results_gptoss_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_2" / "R06_exp3_count_results_gptoss_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "gptoss" / "m_3" / "R06_exp3_count_results_gptoss_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_1" / "R05_exp3_count_results_llama4_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_2" / "R05_exp3_count_results_llama4_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_3" / "R05_exp3_count_results_llama4_m3.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_1" / "R06_exp3_count_results_llama4_m1.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_2" / "R06_exp3_count_results_llama4_m2.json",
    RESULTS_ROOT / "exp3_llmguide" / "llama4" / "m_3" / "R06_exp3_count_results_llama4_m3.json",
]

def main(input_file, output_file):
    # 実験結果の読み出し
    with open(input_file, "r", encoding="utf-8") as f:
        exp_results = json.load(f)
    
    # 問題回答結果の読み出し
    problems = exp_results.get("problems_results", [])

    # 統計情報の初期化
    macro_raw_retrieved_articles_size = 0
    macro_selected_articles_size = 0
    num_problems = 0

    results = []

    # 各問題の情報抽出
    for problem in problems:
        num_problems += 1
        # 問題IDの抽出
        problem_id = problem.get("id", "")
        # 候補条文集合の抽出
        raw_retrieved_articles_ids = (problem.get("first_trial", {})).get("raw_retrieved_articles_set", [])
        raw_retrieved_articles_size = len(raw_retrieved_articles_ids)
        # 選定後の条文分集合の抽出
        selected_articles_ids = (problem.get("second_trial", {})).get("selected_articles", [])
        selected_articles_size = len(selected_articles_ids)
        # 統計情報の更新
        macro_raw_retrieved_articles_size += raw_retrieved_articles_size
        macro_selected_articles_size += selected_articles_size

        # 結果の保存
        results.append({
            "id": problem_id,
            "raw_retrieved_articles_ids": raw_retrieved_articles_ids,
            "raw_retrieved_articles_size": raw_retrieved_articles_size,
            "selected_articles_ids": selected_articles_ids,
            "selected_articles_size": selected_articles_size,
        })
    
    count_output = {
        "summary": {
            "macro_raw_retrieved_articles_size": (macro_raw_retrieved_articles_size / num_problems) if num_problems > 0 else 0,
            "macro_selected_articles_size": (macro_selected_articles_size / num_problems) if num_problems > 0 else 0,
            "num_problems": num_problems,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(count_output, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    for input_file, output_file in zip(INPUT_FILE, OUTPUT_FILE):
        main(input_file, output_file)