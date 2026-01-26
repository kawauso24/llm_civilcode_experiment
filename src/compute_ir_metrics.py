# coding: UTF-8
"""
抽出した条文と正解条文を用いてIR評価指標を計算するモジュール
"""
def compute_metrics(retrieved_articles_set, reference_articles_set):
    true_positives = retrieved_articles_set & reference_articles_set

    precision = len(true_positives) / len(retrieved_articles_set) if retrieved_articles_set else 0.0
    recall = len(true_positives) / len(reference_articles_set) if reference_articles_set else 0.0

    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
    }
