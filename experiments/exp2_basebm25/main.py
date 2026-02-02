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

# OllamaのAPIクライアント設定モジュールをインポート
from ollama_client import create_ollama_model
# コーパスから条文リストを抽出するモジュールをインポート
from load_corpus import extract_articles_from_xml
# テストデータセットから問題を読み出すモジュールをインポート
from load_problems import extract_problems_from_xml
# データセットから正解条文集合を読み出すモジュールをインポート
from load_reference import extract_reference_articles_from_xml
# BM25モデルを構築するモジュールをインポート
from build_bm25model import build_bm25
# BM25を用いて条文を検索するモジュールをインポート
from retrieve_bm25_articles import retrieve_articles
# IR評価指標を計算するモジュールをインポート
from compute_ir_metrics import compute_metrics
# 与えられた条文情報で問題を解くモジュールをインポート
from solve_problems_relevant_articles import solve_problems

# モデル名設定 (ollama版)
MODEL_NAME = [
    "gpt-oss:120b",
    "llama4:128x17b",
]

# 取得する条文の上位件数
M_VALUES = [5, 10, 15]

# 問題データファイル設定
DATASETS = [
    ("R05",
    DATA_ROOT / "coliee_train" / "riteval_R05_jp.xml",
    DATA_ROOT / "coliee_test" / "simple_R05_jp.xml"),
    ("R06",
    DATA_ROOT / "coliee_train" / "riteval_R06_jp.xml",
    DATA_ROOT / "coliee_test" / "simple_R06_jp.xml"),
]

# コーパスファイル設定
CORPUS_FILE = DATA_ROOT / "coliee_corpus" / "civil.xml"

# 結果出力ファイル設定
RESULT_FILE = [
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_5" / "R05_exp2_results_gptoss_m5.json",
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_10" / "R05_exp2_results_gptoss_m10.json",
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_15" / "R05_exp2_results_gptoss_m15.json",
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_5" / "R06_exp2_results_gptoss_m5.json",
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_10" / "R06_exp2_results_gptoss_m10.json",
    RESULTS_ROOT / "exp2_basebm25" / "gptoss" / "m_15" / "R06_exp2_results_gptoss_m15.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_5" / "R05_exp2_results_llama4_m5.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_10" / "R05_exp2_results_llama4_m10.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_15" / "R05_exp2_results_llama4_m15.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_5" / "R06_exp2_results_llama4_m5.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_10" / "R06_exp2_results_llama4_m10.json",
    RESULTS_ROOT / "exp2_basebm25" / "llama4" / "m_15" / "R06_exp2_results_llama4_m15.json",
]

# プロンプト設定
SYSTEM_PROMPT = """
    あなたは日本の法律に詳しい法律家です。
    出力は必ずJSON形式のみで、提示したJSON Schemaに従い、他のテキストは一切含めずに回答してください。
""".strip()
USER_PROMPT = """
    日本の司法試験の民法に関する択一回答式問題です。
    与えられた条文情報のみを利用して記述(t2)が正しいかどうかを判定してください。

    # 参照する条文情報
    {retrieved_articles}
    # 法律に関する記述(t2)
    {t2_text}
""".strip()

# 出力形式の設定(JSON Schema)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "exe2_format",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "description": "記述が正しければY, 間違っていればNと回答してください。",
                    "enum": ["Y", "N"],
                },
            },
            "required": ["judgment"],
            "additionalProperties": False,
        }
    }
}

def main(problem_file, reference_file, result_file, model_name, m):
    # 進捗表示
    print(f"Model: {model_name}, Problem File: {problem_file}, M_Value: {m}")

    # APIクライアント設定
    client = create_ollama_model()

    # テストデータセットから問題を読み出す
    problems = extract_problems_from_xml(problem_file)

    # データセットから正解条文集合を読み出す
    all_reference = extract_reference_articles_from_xml(reference_file)

    # 全条文リストの抽出
    all_articles = extract_articles_from_xml(CORPUS_FILE)

    # BM25モデルの構築
    bm25_model = build_bm25(all_articles)

    problems_results = []
    # 正誤判定に関する統計情報初期化
    num_problems = 0
    num_correct = 0
    num_incorrect = 0
    num_correct_answer_correct_retrieval = 0
    # 条文検索に関する統計情報初期化
    macro_f2 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    num_correct_retrieval = 0

    # 各問題に対して検索・推論を実行
    for problem in problems:
        try:
            # 問題数のカウントアップ
            num_problems += 1

            # 各問題の情報抽出
            problem_id = problem.get("id")
            correct_label = problem.get("correct_label")
            reference_text = problem["t1"]
            problem_text = problem["t2"]

            # BM25を用いて関連条文上位m件を取得
            retrieved_articles = retrieve_articles(
                bm25_model,
                all_articles,
                problem_text,
                m,
                source="BM25",
            )
            
            # 正解条文集合を取得
            reference_article_nums = all_reference[problem_id]
            # print(reference_article_nums)

            # 抽出した条文の条文集合を取得
            retrieved_article_nums = set()
            for article in retrieved_articles:
                article_num = article["article"]["num"]
                retrieved_article_nums.add(article_num)
            # print(retrieved_article_nums)
            
            # IR評価の計算
            ir_metrics = compute_metrics(retrieved_article_nums, reference_article_nums)
            precision = ir_metrics["precision"]
            recall = ir_metrics["recall"]
            f2 = ir_metrics["f2"]

            # 統計情報の更新
            macro_f2 += f2
            macro_precision += precision
            macro_recall += recall

            # 正解条文をすべて抽出できたかどうか判定
            is_correct_retrieval = (reference_article_nums.issubset(retrieved_article_nums))
            if is_correct_retrieval:
                num_correct_retrieval += 1

            # 関連条文を与えてLLMに問題を解かせる
            model_output = solve_problems(
                client=client,
                model_name=model_name,
                retrieved_articles=retrieved_articles,
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
                if is_correct_retrieval:
                    num_correct_answer_correct_retrieval += 1
            else:
                num_incorrect += 1
            
            # 結果の保存
            problems_results.append({
                "id": problem_id,
                "correct_label": correct_label,
                "model_output": model_output,
                "retrieved_articles": retrieved_articles,
                "correct": is_correct,
                "reference_text": reference_text,
                "problem_text": problem_text,
            })

            # 進捗表示
            print(f"Processed Problem ID: {problem_id}")
        except Exception as e:
            print(f"Error processing Problem ID: {problem_id}, Error: {e}")
    
    # 集計結果を含めた最終出力の保存
    results = {
        "judge_summary": {
            "num_problems": num_problems,
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "accuracy": ((num_correct / num_problems) * 100) if num_problems > 0 else 0.0,
            "num_correct_answer_correct_retrieval": num_correct_answer_correct_retrieval,
            "num_correct_retrieval": num_correct_retrieval,
            "correct_answer_correct_retrieval_rate": ((num_correct_answer_correct_retrieval / num_correct_retrieval) * 100) if num_correct_retrieval > 0 else 0.0,
        },
        "retrieval_summary": {
            "macro_f2": round((macro_f2 / num_problems) if num_problems > 0 else 0.0, 4),
            "macro_precision": round((macro_precision / num_problems) if num_problems > 0 else 0.0, 4),
            "macro_recall": round((macro_recall / num_problems) if num_problems > 0 else 0.0, 4),
        },
        "problems_results": problems_results,
    }

    # 結果をJSONファイルに保存
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    idx = 0
    for model_name in MODEL_NAME:
        for dataset_id, problem_file, reference_file in DATASETS:
            for m in M_VALUES:
                result_file = RESULT_FILE[idx]
                idx += 1
                main(problem_file, reference_file, result_file, model_name,m)
