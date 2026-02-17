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
# ゼロショットで問題を解くモジュールをインポート
from solve_problems_zeroshot import solve_zeroshot_problems

# モデル名設定
MODEL_NAME = "llama4:128x17b"

# ベースで取得する条文数設定
BASE_QUERY_NUM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# クエリ拡張のためにさらに取得する条文数設定
QUERY_EXPAND_NUM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 問題データセットの設定
PROBLEM_FILE = DATA_ROOT / "coliee_train" / "riteval_R06_jp.xml"
REFERENCE_FILE = DATA_ROOT / "coliee_test" / "simple_R06_jp.xml"

# コーパスファイルの設定
CORPUS_FILE = DATA_ROOT / "coliee_corpus" / "civil.xml"

# 準用マッピング設定
JUNYO_FILE = SRC_ROOT / "junyo_mapping.json"

# 出力ファイル設定
OUTPUT_FILE = RESULTS_ROOT / "parameter_search_results.jsonl"

# プロンプト設定
SYSTEM_PROMPT = """
    あなたは日本の法律に詳しい法律家です。
    なお、条文を陳述する際には条文番号のみではなく条文全文を引用すること。
    出力は必ずJSON形式のみで、提示したJSON Schemaに従い、他のテキストは一切含めずに回答してください。
""".strip()

USER_PROMPT_QUERY_EXPAND = """
    日本の司法試験の民法に関する択一回答式問題です。
    記述(t2)が正しいかどうかを判定し、その回答根拠となる民法条文の該当箇所を引用して示してください。

    # 法律に関する記述(t2)
    {t2_text}
""".strip()

USER_PROMPT_SELECT = """
    日本の司法試験の民法に関する択一回答式問題です。
    与えられた条文情報のみを利用して記述(t2)が正しいかどうかを判定してください。
    また実際に判定に利用した条文番号を、与えられた参照条文番号のリストから選んで回答してください。

    # 参照する条文情報
    {retrieved_articles}
    # 参照する条文番号リスト
    {retrieved_article_ids}
    # 法律に関する記述(t2)
    {t2_text}
""".strip()

USER_PROMPT_ENTAILMENT = """
    日本の司法試験の民法に関する択一回答式問題です。
    与えられた条文情報のみを利用して記述(t2)が正しいかどうかを判定してください。

    # 参照する条文情報
    {retrieved_articles}
    # 法律に関する記述(t2)
    {t2_text}
""".strip()

# 出力形式の設定(JSON Schema)
RESPONSE_FORMAT_QUERY_EXPAND = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_expand_format",
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
                    "type": "string",
                    "minLength": 20,    # 最低20文字以上にすることで、"民法◯条"のような不十分な回答を防ぐ
                    "description": "判断の根拠となった法律の条文全文を（第◯条 ①~ ②~）のように引用。民法以外は記載しないこと。",
                },
            },
            "required": ["judgment", "reason_articles"],
            "additionalProperties": False,
        }
    }
}

# Selectorに渡す出力形式を定義する関数
def return_response_schema(articles_ids_set):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "selector_format",
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
                        "description": "与えられた条文番号リストの中から実際に回答に利用した条文番号を回答してください.",
                        "minItems": 1,  # 最小1件以上を強制
                        "items": {
                            "type": "integer",
                            "enum": list(articles_ids_set)
                        },
                    },
                },
                "required": ["judgment", "selected_articles"],
                "additionalProperties": False,
            }
        }
    }

RESPONSE_FORMAT_ENTAILMENT = {
    "type": "json_schema",
    "json_schema": {
        "name": "entailment_format",
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

def main(query_expand_num, base_query_num):
    # 進捗表示
    print(f"Running with query_expand_num={query_expand_num}, base_query_num={base_query_num}")

    # OllamaのAPIクライアントを作成
    client = create_ollama_model()

    # コーパスXMLから条文リストを抽出 (dictのlist, 各dictのキー: 条文番号, 値: 条文本文)
    all_articles = extract_articles_from_xml(CORPUS_FILE)

    # BM25モデルを構築
    bm25_model = build_bm25(all_articles)

    # 問題データセットXMLから問題リストを抽出 (dictのlist、各dictのキー: 問題ID, 正解ラベル, t1, t2, 各dictの値: それぞれの値)
    problems = extract_problems_from_xml(PROBLEM_FILE)

    # データセットから正解条文集合を読み出す (dict: キー: 問題ID, 値: 正解条文番号のset)
    all_reference = extract_reference_articles_from_xml(REFERENCE_FILE)

    # 準用mappingの読み出し
    with open(JUNYO_FILE, "r", encoding="utf-8") as f:
        junyo_mapping = json.load(f)

    # BM25モデルの構築
    bm25_model = build_bm25(all_articles)

    problems_results = []
    max_f2 = 0.0
    max_query_expand_num = 0
    max_base_query_num = 0

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
    macro_bm25 = 0.0
    macro_llmoutput = 0.0
    macro_junyo = 0.0
    # 条文検索に関する統計情報初期化 (2回目)
    macro_second_f2 = 0.0
    macro_second_precision = 0.0
    macro_second_recall = 0.0
    num_second_correct_retrieval = 0

    # 各問題に対して検索・推論を実行
    for problem in problems:
        try:
            # 各問題に対する条文ソース別正解条文数カウント初期化
            num_basebm25 = 0
            num_llmguide = 0
            num_junyo = 0

            # 問題数のカウントアップ
            num_problems += 1

            # 各問題の情報抽出
            problem_id = problem.get("id")
            correct_label = problem.get("correct_label")
            reference_text = problem["t1"]
            problem_text = problem["t2"]
            reference_article_nums = all_reference[problem_id]
            print(f"reference_article_nums: {reference_article_nums}")

            # まずはzeroshotで推論し、query expansionに利用する条文を抽出
            zeroshot_result = solve_zeroshot_problems(
                client,
                MODEL_NAME,
                problem_text,
                SYSTEM_PROMPT,
                USER_PROMPT_QUERY_EXPAND,
                RESPONSE_FORMAT_QUERY_EXPAND
            )
            model_reason = zeroshot_result.get("reason_articles", "")
            print(f"Zeroshot reason articles: {model_reason}")

            # BM25を用いて問題文による検索結果上位m件を取得
            base_retrieved_articles = retrieve_articles(
                bm25_model,
                all_articles,
                problem_text,
                m=base_query_num,
                source="BM25",
            )
            base_retrieved_article_ids = set([article["article"]["num"] for article in base_retrieved_articles])
            print(f"Base retrieved articles set: {base_retrieved_article_ids}")

            # LLMの出力をクエリとした条文検索を行う
            llmoutput_retrieved_articles = retrieve_articles(
                bm25_model,
                all_articles,
                model_reason,
                m=query_expand_num,
                source="LLMoutput",
            )
            llmoutput_retrieved_article_ids = set([article["article"]["num"] for article in llmoutput_retrieved_articles])
            print(f"LLM output retrieved articles set: {llmoutput_retrieved_article_ids}")

            # 問題文による検索+LLM出力による検索の組み合わせで取得した条文リストに準用条文を追加
            raw_retrieved_articles = []
            raw_retrieved_articles_set = set()
            (raw_retrieved_articles, raw_retrieved_articles_set) = add_junyo(
                base_retrieved_articles,
                llmoutput_retrieved_articles,
                junyo_mapping,
                all_articles,
            )
            print(f"Raw retrieved articles set: {raw_retrieved_articles_set}")

            # 抽出した条文のソースをカウント
            for article in raw_retrieved_articles:
                source = article.get("source", "")
                article_num = article["article"]["num"]
                if (source == "BM25") and (article_num in reference_article_nums):
                    num_basebm25 += 1
                elif (source == "LLMoutput") and (article_num in reference_article_nums):
                    num_llmguide += 1
                elif (source == "Junyo") and (article_num in reference_article_nums):
                    num_junyo += 1
            
            # IR評価の計算 (1回目)
            first_ir_metrics = compute_metrics(raw_retrieved_articles_set, reference_article_nums)
            first_precision = first_ir_metrics["precision"]
            first_recall = first_ir_metrics["recall"]
            first_f2 = first_ir_metrics["f2"]

            # 統計情報の更新 (1回目)
            macro_first_f2 += first_f2
            macro_first_precision += first_precision
            macro_first_recall += first_recall
            macro_bm25 += (num_basebm25 / len(reference_article_nums)) if len(reference_article_nums) > 0 else 0.0
            macro_llmoutput += (num_llmguide / len(reference_article_nums)) if len(reference_article_nums) > 0 else 0.0
            macro_junyo += (num_junyo / len(reference_article_nums)) if len(reference_article_nums) > 0 else 0.0

            # 正解条文をすべて抽出できたかどうか判定 (1回目)
            is_first_correct_retrieval = (reference_article_nums.issubset(raw_retrieved_articles_set))
            if is_first_correct_retrieval:
                num_first_correct_retrieval += 1

            # 問題を解く + 必要条文の選択
            model_first_output = solve_problems(
                client,
                MODEL_NAME,
                raw_retrieved_articles,
                problem_text,
                SYSTEM_PROMPT,
                USER_PROMPT_SELECT.format(
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
            print(f"Selected articles set: {selected_articles_set}")

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
                MODEL_NAME,
                selected_articles,
                problem_text,
                SYSTEM_PROMPT,
                USER_PROMPT_ENTAILMENT.format(
                    retrieved_articles=build_context_from_articles(selected_articles),
                    t2_text=problem_text,
                ),
                RESPONSE_FORMAT_ENTAILMENT,
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

            # 進捗表示
            print(f"Processed Problem ID: {problem_id}")
        except Exception as e:
            print(f"Error processing Problem ID: {problem_id}, Error: {e}")
    
    # 集計結果
    results = {
        "query_expand_num": query_expand_num,
        "base_query_num": base_query_num,
        "num_problems": num_problems,
        "retrieval_summary": {
            "macro_f2": round((macro_second_f2 / num_problems) if num_problems > 0 else 0.0, 4),
            "macro_precision": round((macro_second_precision / num_problems) if num_problems > 0 else 0.0, 4),
            "macro_recall": round((macro_second_recall / num_problems) if num_problems > 0 else 0.0, 4),
            "source_coverage": {
                "BM25": round((macro_bm25 / num_problems) if num_problems > 0 else 0.0, 4),
                "LLMoutput": round((macro_llmoutput / num_problems) if num_problems > 0 else 0.0, 4),
                "Junyo": round((macro_junyo / num_problems) if num_problems > 0 else 0.0, 4),
            },
        }
    }
    if results["retrieval_summary"]["macro_f2"] > max_f2:
        max_f2 = results["retrieval_summary"]["macro_f2"]
        max_query_expand_num = query_expand_num
        max_base_query_num = base_query_num

    
    # 結果をファイルに保存
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for i in range(3):
        # ベースで取得する条文数とクエリ拡張のためにさらに取得する条文数の組み合わせを全て試す
        for base_query_num in BASE_QUERY_NUM:
            for query_expand_num in QUERY_EXPAND_NUM:
                main(query_expand_num, base_query_num)
        # 全ての組み合わせの実験が終わったら、結果ファイルを読み込んで最もF2スコアが高かった組み合わせを表示する
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "max_f2": max_f2,
                "max_query_expand_num": max_query_expand_num,
                "max_base_query_num": max_base_query_num,
            }, ensure_ascii=False) + "\n")
