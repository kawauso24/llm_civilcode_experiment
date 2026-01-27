# coding: UTF-8
"""
BM25モデルを用いて条文を検索するモジュール
"""
from build_bm25model import tokenize

def retrieve_articles(bm25, all_articles, query, m, source):
    # クエリのトークン化
    query_tokens = tokenize(query)

    # BM25によるスコア計算
    scores = bm25.get_scores(query_tokens)

    # スコアの高い順にソートして上位m件を取得
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:m]

    results = []

    for index in ranked_indices:
        results.append({
            "article": all_articles[index],
            "score": scores[index],
            "source": source,
        })
    
    return results