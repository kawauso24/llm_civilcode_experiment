# coding: UTF-8
"""
ゼロショットで問題を解くためのモジュール
"""
import json
from openai import OpenAI

def solve_problems(client, model_name, problem_text, system_prompt, user_prompt, response_format):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                t2_text=problem_text
            )},
        ],
        temperature=0.0,
        response_format=response_format,
    )
    model_output = response.choices[0].message.content
    print(model_output)
    return json.loads(model_output)
