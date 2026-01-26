# coding: UTF-8
"""
LLMに法律問題を解かせる際に、関連条文を渡して解かせるモジュール
"""
import json
from openai import OpenAI

def build_context_from_articles(retrieved_articles):
    chunks = []

    for article in retrieved_articles:
        num = article['article']['num']
        text = article['article']['text']

        header = f"【民法第{num}条\n"
        chunk = f"{header}\n{text}\n"
        chunks.append(chunk)

    # 条文ごとに空行で区切る
    return '\n\n'.join(chunks)

def solve_problems(client, model_name, retrieved_articles, problem_text, system_prompt, user_prompt, response_format):
    # LLMに渡す用に条文リストを整形
    context = build_context_from_articles(retrieved_articles)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                retrieved_articles=context,
                t2_text=problem_text
            )},
        ],
        temperature=0.0,
        response_format=response_format,
    )
    model_output = response.choices[0].message.content
    print(model_output)
    return json.loads(model_output)