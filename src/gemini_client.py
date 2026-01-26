# coding: UTF-8
"""
Gemini APIを利用するためのクライアント設定モジュール
"""
from google import genai
import os
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

# APIクライアント設定
def create_gemini_model_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client