# coding: UTF-8
"""
コーパスからBM25モデルを構築するモジュール
"""
from fugashi import Tagger
from rank_bm25 import BM25Okapi

# 形態素解析器の初期化
tagger = Tagger()

# 入力テキストをトークン列に変換する関数
def tokenize(text):
    return [word.surface for word in tagger(text)]

# BM25モデルを構築する関数 (文書のトークン化・各単語の頻度情報(DF)の計算・文書長や平均文書長の記録)
def build_bm25(articles):
    # 各条文テキストのトークン化
    corpus_tokens = [tokenize(article['text']) for article in articles]

    # BM25モデルの構築
    bm25 = BM25Okapi(corpus_tokens)

    return bm25