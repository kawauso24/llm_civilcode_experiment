# coding: UTF-8
"""
モデルが出力した根拠条文をGeminiに評価させるモジュール
"""
def eval_reason(client, model, model_reason, correct_reason, system_prompt, user_prompt, response_schema):
    # プロンプトをまとめる
    prompt = user_prompt.format(
        t1_text=model_reason,
        t2_text=correct_reason,
        schema=json.dumps(response_schema, ensure_ascii=False),
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": response_schema,
            "temperature": 0.0,
        },
    )

    print("Gemini Response Content:", response.parsed)
    return response.parsed