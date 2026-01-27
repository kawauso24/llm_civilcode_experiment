# coding: UTF-8
"""
問題文をクエリとしてBM25で条文検索した結果と、LLM出力をクエリとしてBM25で条文検索した結果をマージして、さらに準用条文を加えてlistで返すモジュール
"""
def add_junyo(base_retrieved_articles, llmoutput_retrieved_articles, junyo_mapping, all_articles):
    # 問題文による検索結果とLLM出力による検索結果をマージする
    (base_llmoutput_articles, base_llmoutput_articles_set) = merge_bm25_llmoutput(
        base_retrieved_articles, 
        llmoutput_retrieved_articles,
    )

    # 条文番号から条文テキストを取得するための辞書を作成
    article_text_by_num = {a["num"]: a["text"] for a in all_articles}

    # 準用の順方向逆方向の対応をそれぞれ取得
    forward = junyo_mapping.get("forward", {})
    backward = junyo_mapping.get("backward", {})

    # 検索結果をまとめる変数を定義
    raw_retrieved_articles = base_llmoutput_articles
    raw_retrieved_articles_set = base_llmoutput_articles_set

    for article_num in list(base_llmoutput_articles_set):
        # 順方向の準用条文を追加
        if article_num in forward:
            for junyo_num in forward[article_num]:
                if junyo_num not in base_llmoutput_articles_set:
                    junyo_article = article_text_by_num.get(junyo_num)
                    if junyo_article is not None:
                        raw_retrieved_articles.append({
                            "article": {
                                "num": junyo_num,
                                "text": junyo_article,
                            },
                            "source": "Junyo",
                        })
                        raw_retrieved_articles_set.add(junyo_num)
        # 逆方向の準用条文を追加
        if article_num in backward:
            for junyo_num in backward[article_num]:
                if junyo_num not in base_llmoutput_articles_set:
                    junyo_article = article_text_by_num.get(junyo_num)
                    if junyo_article is not None:
                        raw_retrieved_articles.append({
                            "article": {
                                "num": junyo_num,
                                "text": junyo_article,
                            },
                            "source": "Junyo",
                        })
                        raw_retrieved_articles_set.add(junyo_num)
    
    return (raw_retrieved_articles, raw_retrieved_articles_set)



# BM25での問題文による検索結果とLLM出力による検索結果をマージする関数
def merge_bm25_llmoutput(base_retrieved_articles, llmoutput_retrieved_articles):
    base_llmoutput_articles = []
    base_llmoutput_articles_set = set()
    
    for article in base_retrieved_articles:
        base_llmoutput_articles.append(article)
        base_llmoutput_articles_set.add(article["article"]["num"])
    for article in llmoutput_retrieved_articles:
        if article["article"]["num"] not in base_llmoutput_articles_set:
            base_llmoutput_articles.append(article)
            base_llmoutput_articles_set.add(article["article"]["num"])
    
    return (base_llmoutput_articles, base_llmoutput_articles_set)