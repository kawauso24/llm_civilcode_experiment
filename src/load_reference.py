# coding: UTF-8
"""
データセットから正解条文を抽出してdictとして返すモジュール
"""
import xml.etree.ElementTree as ET

def extract_reference_articles_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    reference_articles = {}

    for problem in root.findall("pair"):
        problem_id = problem.get("id")

        articles = {
            article.text.strip()
            for article in problem.findall("article")
            if article.text is not None
        }

        reference_articles[problem_id] = articles

    return reference_articles