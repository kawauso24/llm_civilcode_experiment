# coding: UTF-8
"""
テストデータセットから問題を抽出してリストとして返すモジュール
"""
import xml.etree.ElementTree as ET

def extract_problems_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    problems = []
    for problem in root.findall("pair"):
        problems.append({
            "id": problem.get("id"),    # 各問題のID取得
            "correct_label": problem.get("label"),  # 各問題の正解ラベル取得
            "t1": (problem.findtext("t1") or "").strip(),  # 参考条文取得と前後の空白削除
            "t2": (problem.findtext("t2") or "").strip(),  # 問題文取得と前後の空白削除
        })

    return problems