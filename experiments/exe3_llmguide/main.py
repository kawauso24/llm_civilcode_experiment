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

# LLM outputのファイル設定
LLM_OUTPUT_FILE = [
    RESULTS_FILE / "exe1_zeroshot" / "gptoss" / "R05_exe1_results_gptoss.json",
    RESULTS_FILE / "exe1_zeroshot" / "gptoss" / "R06_exe1_results_gptoss.json",
    RESULTS_FILE / "exe1_zeroshot" / "llama4" / "R05_exe1_results_llama4.json",
    RESULTS_FILE / "exe1_zeroshot" / "llama4" / "R06_exe1_results_llama4.json",
]

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
USER_PROMPT = """
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

# 出力形式の設定(JSON Schema)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "exe3_format",
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
                    "description": "参照した条文番号のリストを配列で回答してください。",
                    "items": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
            },
            "required": ["judgment", "selected_articles"],
            "additionalProperties": False,
        }
    }
}

def main(problem_file, reference_file, result_file, llmoutput_file, model_name, m):
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