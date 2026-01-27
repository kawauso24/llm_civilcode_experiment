# coding: UTF-8
"""
実験1で出力したLLMの出力した根拠条文をJSONファイルから抽出し、dict(キー: 問題ID, 値: LLM出力)で返すモジュール
"""
import json

def extract_llmreason_from_json(llmoutput_file):
    llm_reasons = {}
    with open(llmoutput_file, "r", encoding="utf-8") as f:
        llm_outputs = json.load(f)
        for problem in llm_outputs.get("problems_results", []):
            problem_id = problem.get("id")
            llmoutput = (problem.get("model_output", {})).get("reason_articles", "").strip()
            llm_reasons[problem_id] = llmoutput
    return llm_reasons