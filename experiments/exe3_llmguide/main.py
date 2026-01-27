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
# 実験1の出力ファイルからLLMの根拠条文を抽出するモジュールをインポート
from load_llmreason import extract_llmreason_from_json
# BM25モデルを構築するモジュールをインポート
from build_bm25model import build_bm25
# BM25を用いて条文を検索するモジュールをインポート
from retrieve_bm25_articles import retrieve_articles
# 問題文とLLM出力それぞれでBM25検索をした結果をマージして、さらに準用条文を加えるモジュールをインポート
from add_junyo_articles import add_junyo
# IR評価指標を計算するモジュールをインポート
from compute_ir_metrics import compute_metrics
# 与えられた条文情報で問題を解くモジュールをインポート
from solve_problems_relevant_articles import solve_problems, build_context_from_articles

# モデル名設定 (ollama版)
MODEL_NAME = [
    "gpt-oss:120b",
    "llama4:128x17b",
]

# 取得する条文の上位件数
M_VALUES = [1, 2, 3]

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

# LLM outputのファイル設定
LLM_OUTPUT_FILE = {
    "R05_gptoss": RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R05_exe1_results_gptoss.json",
    "R06_gptoss": RESULTS_ROOT / "exe1_zeroshot" / "gptoss" / "R06_exe1_results_gptoss.json",
    "R05_llama4": RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R05_exe1_results_llama4.json",
    "R06_llama4": RESULTS_ROOT / "exe1_zeroshot" / "llama4" / "R06_exe1_results_llama4.json",
}

# 準用マッピングファイル設定
JUNYO_FILE = SRC_ROOT / "junyo_mapping.json"

# 結果出力ファイル設定
RESULT_FILE = [
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_1" / "R05_exe3_results_gptoss_m1.json",
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_2" / "R05_exe3_results_gptoss_m2.json",
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_3" / "R05_exe3_results_gptoss_m3.json",
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_1" / "R06_exe3_results_gptoss_m1.json",
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_2" / "R06_exe3_results_gptoss_m2.json",
    RESULTS_ROOT / "exe3_llmguide" / "gptoss" / "m_3" / "R06_exe3_results_gptoss_m3.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_1" / "R05_exe3_results_llama4_m1.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_2" / "R05_exe3_results_llama4_m2.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_3" / "R05_exe3_results_llama4_m3.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_1" / "R06_exe3_results_llama4_m1.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_2" / "R06_exe3_results_llama4_m2.json",
    RESULTS_ROOT / "exe3_llmguide" / "llama4" / "m_3" / "R06_exe3_results_llama4_m3.json",
]

# プロンプト設定
SYSTEM_PROMPT = """
    あなたは日本の法律に詳しい法律家です。
    出力は必ずJSON形式のみで、提示したJSON Schemaに従い、他のテキストは一切含めずに回答してください。
""".strip()
USER_PROMPT_FIRST = """
    日本の司法試験の民法に関する択一回答式問題です。
    与えられた条文情報のみを利用して記述(t2)が正しいかどうかを判定してください。
    また実際に判定に利用した条文番号を与えた条文番号リストの中から選んでリストとして回答してください。

    # 参照する条文情報
    {retrieved_articles}
    # 参照する条文番号リスト
    {retrieved_article_ids}
    # 法律に関する記述(t2)
    {t2_text}
""".strip()
USER_PROMPT_SECOND = """
    日本の司法試験の民法に関する択一回答式問題です。
    与えられた条文情報のみを利用して記述(t2)が正しいかどうかを判定してください。

    # 参照する条文情報
    {retrieved_articles}
    # 法律に関する記述(t2)
    {t2_text}
""".strip()

# 1回目の推論時の出力形式の設定(JSON Schema)
def return_response_schema(articles_ids_set):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "exe3_first_format",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "judgment": {
                        "type": "string",
                        "description": "記述が正しければY, 間違っていればNと回答してください。",
                        "enum": ["Y", "N"],
                    },
                    "selected_articles": {
                        "type": "array",
                        "description": "参照した条文番号のリストをarrayで回答してください.",
                        "items": {
                            "type": "integer",
                            "minItems": 1,
                            "enum": list(articles_ids_set)
                        },
                    },
                },
                "required": ["judgment", "selected_articles"],
                "additionalProperties": False,
            }
        }
    }

# 2回目の推論時の出力形式の設定(JSON Schema)
RESPONSE_FORMAT_SECOND = {
    "type": "json_schema",
    "json_schema": {
        "name": "exe3_second_format",
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


def main(problem_file, reference_file, result_file, llmoutput_file, model_name, m):
    # 進捗表示
    print(f"Model: {model_name}, Problem File: {problem_file}, M_Value: {m}, LLM Output File: {llmoutput_file}")

    # APIクライアント設定
    client = create_ollama_model()

    # テストデータセットから問題を読み出す (dictのlist、各dictのキー: 問題ID, 正解ラベル, t1, t2, 各dictの値: それぞれの値)
    problems = extract_problems_from_xml(problem_file)

    # データセットから正解条文集合を読み出す (dict: キー: 問題ID, 値: 正解条文番号のset)
    all_reference = extract_reference_articles_from_xml(reference_file)

    # 全条文リストの抽出 (dictのlist, 各dictのキー: 条文番号, 値: 条文本文)
    all_articles = extract_articles_from_xml(CORPUS_FILE)

    # 各問題に対してLLM出力を読み出す (dict: キー: 問題ID, 値: LLM出力)
    llmoutput_reasons = extract_llmreason_from_json(llmoutput_file)

    # 準用mappingの読み出し
    with open(JUNYO_FILE, "r", encoding="utf-8") as f:
        junyo_mapping = json.load(f)

    # BM25モデルの構築
    bm25_model = build_bm25(all_articles)

    problems_results = []
    # 正誤判定に関する統計情報初期化 (1回目)
    num_problems = 0
    num_first_correct = 0
    num_first_incorrect = 0
    num_first_correct_answer_correct_retrieval = 0
    # 正誤判定に関する統計情報初期化 (2回目)
    num_second_correct = 0
    num_second_incorrect = 0
    num_second_correct_answer_correct_retrieval = 0
    # 条文検索に関する統計情報初期化 (1回目)
    macro_first_f2 = 0.0
    macro_first_precision = 0.0
    macro_first_recall = 0.0
    num_first_correct_retrieval = 0
    num_basebm25 = 0
    num_llmguide = 0
    num_junyo = 0
    # 条文検索に関する統計情報初期化 (2回目)
    macro_second_f2 = 0.0
    macro_second_precision = 0.0
    macro_second_recall = 0.0
    num_second_correct_retrieval = 0

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

            # BM25を用いて問題文による検索結果上位5件を取得
            base_retrieved_articles = retrieve_articles(
                bm25_model,
                all_articles,
                problem_text,
                m=5,
                source="BM25",
            )

            # LLMの根拠条文出力を取得
            llm_reason = (llmoutput_reasons.get(problem_id, "")).strip()

            # BM25を用いてLLM出力による検索結果上位m件を取得
            llmoutput_retrieved_articles = retrieve_articles(
                bm25_model,
                all_articles,
                llm_reason,
                m,
                source="LLMoutput",
            )

            # 問題文による検索+LLM出力による検索の組み合わせで取得した条文リストに準用条文を追加
            raw_retrieved_articles = []
            raw_retrieved_articles_set = set()
            (raw_retrieved_articles, raw_retrieved_articles_set) = add_junyo(
                base_retrieved_articles,
                llmoutput_retrieved_articles,
                junyo_mapping,
                all_articles,
            )
            print(raw_retrieved_articles_set)

            # 正解条文集合を取得
            reference_article_nums = all_reference[problem_id]

            # IR評価の計算 (1回目)
            first_ir_metrics = compute_metrics(raw_retrieved_articles_set, reference_article_nums)
            first_precision = first_ir_metrics["precision"]
            first_recall = first_ir_metrics["recall"]
            first_f2 = first_ir_metrics["f2"]

            # 統計情報の更新 (1回目)
            macro_first_f2 += first_f2
            macro_first_precision += first_precision
            macro_first_recall += first_recall

            # 正解条文をすべて抽出できたかどうか判定 (1回目)
            is_first_correct_retrieval = (reference_article_nums.issubset(raw_retrieved_articles_set))
            if is_first_correct_retrieval:
                num_first_correct_retrieval += 1
            
            # 問題を解く + 必要条文の選択
            model_first_output = solve_problems(
                client,
                model_name,
                raw_retrieved_articles,
                problem_text,
                SYSTEM_PROMPT,
                USER_PROMPT_FIRST.format(
                    retrieved_article_ids=list(raw_retrieved_articles_set),
                    retrieved_articles=build_context_from_articles(raw_retrieved_articles),
                    t2_text=problem_text,
                ),
                return_response_schema(raw_retrieved_articles_set),
            )

            # 正誤判定 (1回目)
            is_first_correct = (model_first_output.get("judgment") == correct_label)

            # 正解数のカウントアップ (1回目)
            if is_first_correct:
                num_first_correct += 1
                if is_first_correct_retrieval:
                    num_first_correct_answer_correct_retrieval += 1
            else:
                num_first_incorrect += 1

            # モデルが実際に使った条文番号リストを取得
            selected_articles_set = set(model_first_output.get("selected_articles", []))
            print(selected_articles_set)

            # モデルが選んだ条文リストから条文情報を取得
            selected_articles = []
            for article in raw_retrieved_articles:
                article_num = article["article"]["num"]
                if article_num in selected_articles_set:
                    selected_articles.append(article)
            
            # IR評価の計算 (2回目)
            second_ir_metrics = compute_metrics(selected_articles_set, reference_article_nums)
            second_precision = second_ir_metrics["precision"]
            second_recall = second_ir_metrics["recall"]
            second_f2 = second_ir_metrics["f2"]

            # 統計情報の更新 (2回目)
            macro_second_f2 += second_f2
            macro_second_precision += second_precision
            macro_second_recall += second_recall

            # 正解条文をすべて抽出できたかどうか判定 (2回目)
            is_second_correct_retrieval = (reference_article_nums.issubset(selected_articles_set))
            if is_second_correct_retrieval:
                num_second_correct_retrieval += 1

            # 問題を解く (2回目)
            model_second_output = solve_problems(
                client,
                model_name,
                selected_articles,
                problem_text,
                SYSTEM_PROMPT,
                USER_PROMPT_SECOND,
                RESPONSE_FORMAT_SECOND,
            )

            # 正誤判定 (2回目)
            is_second_correct = (model_second_output.get("judgment") == correct_label)

            # 正解数のカウントアップ (2回目)
            if is_second_correct:
                num_second_correct += 1
                if is_second_correct_retrieval:
                    num_second_correct_answer_correct_retrieval += 1
            else:
                num_second_incorrect += 1

            # 結果の保存
            problems_results.append({
                "id": problem_id,
                "correct_label": correct_label,
                "reference_text": reference_text,
                "problem_text": problem_text,
                "first_trial": {
                    "model_output": model_first_output,
                    "correct": is_first_correct,
                    "raw_retrieved_articles": raw_retrieved_articles,
                    "raw_retrieved_articles_set": raw_retrieved_articles_set,
                },
                "second_trial": {
                    "model_output": model_second_output,
                    "correct": is_second_correct,
                    "selected_articles": selected_articles,
                    "selected_articles_set": selected_articles_set,
                }
            })

            # 進捗表示
            print(f"Processed Problem ID: {problem_id}")
        except Exception as e:
            print(f"Error processing Problem ID: {problem_id}, Error: {e}")

    # 集計結果を含めた最終出力の保存
    results = {
        "first_trial": {
            "judge_summary": {
                "num_problems": num_problems,
                "num_correct": num_first_correct,
                "num_incorrect": num_first_incorrect,
                "accuracy": ((num_first_correct / num_problems) * 100.0) if num_problems > 0 else 0.0,
                "num_correct_answer_correct_retrieval": num_first_correct_answer_correct_retrieval,
                "correct_answer_correct_retrieval_rate": ((num_first_correct_answer_correct_retrieval / num_first_correct_retrieval) * 100.0) if num_first_correct_retrieval > 0 else 0.0,
            },
            "retrieval_summary": {
                "macro_f2": round((macro_first_f2 / num_problems) if num_problems > 0 else 0.0, 4),
                "macro_precision": round((macro_first_precision / num_problems) if num_problems > 0 else 0.0, 4),
                "macro_recall": round((macro_first_recall / num_problems) if num_problems > 0 else 0.0, 4),
            },
        },
        "second_trial": {
            "judge_summary": {
                "num_problems": num_problems,
                "num_correct": num_second_correct,
                "num_incorrect": num_second_incorrect,
                "accuracy": ((num_second_correct / num_problems) * 100.0) if num_problems > 0 else 0.0,
                "num_correct_answer_correct_retrieval": num_second_correct_answer_correct_retrieval,
                "correct_answer_correct_retrieval_rate": ((num_second_correct_answer_correct_retrieval / num_second_correct_retrieval) * 100.0) if num_second_correct_retrieval > 0 else 0.0,
            },
            "retrieval_summary": {
                "macro_f2": round((macro_second_f2 / num_problems) if num_problems > 0 else 0.0, 4),
                "macro_precision": round((macro_second_precision / num_problems) if num_problems > 0 else 0.0, 4),
                "macro_recall": round((macro_second_recall / num_problems) if num_problems > 0 else 0.0, 4),
            },
        },
        "problems_results": problems_results,
    }

    # 結果をファイルに保存
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    idx = 0
    for model_name in MODEL_NAME:
        if model_name == "gpt-oss:120b":
            model_key_prefix = "gptoss"
        else: 
            model_key_prefix = "llama4"
        for dataset_id, problem_file, reference_file in DATASETS:
            llmoutput_file = LLM_OUTPUT_FILE[f"{dataset_id}_{model_key_prefix}"]
            for m in M_VALUES:
                result_file = RESULT_FILE[idx]
                idx += 1
                main(
                    problem_file,
                    reference_file,
                    result_file,
                    llmoutput_file,
                    model_name,
                    m,
                )