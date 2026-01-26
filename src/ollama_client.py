# coding: utf-8
"""
OllamaのAPIクライアント設定を行うモジュール
"""
from openai import OpenAI
import os
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

# APIクライアント設定
def create_ollama_model():
    client = OpenAI(
        base_url=os.getenv("OLLAMA_URL"),   # ローカルネットワーク下
        # base_url=os.getenv("SERVER_LLM_URL"),  # 別のサーバーに入ってMacを使う場合
        api_key=os.getenv("OLLAMA_API_KEY")
    )
    return client