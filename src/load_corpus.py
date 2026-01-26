# coding: UTF-8
"""
コーパスXMLから条文リストを抽出するモジュール
"""
from pathlib import Path
from xml.etree import ElementTree as ET

# XMLファイルから条文リストを作成する関数
def extract_articles_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # LawBodyを探す(e-Govコーパス形式)
    # law_body = root.find('.//LawBody')

    articles = []

    # メインの条文を抽出(e-Govコーパス形式)
    # for article in law_body.findall(".//MainProvision//Article"):
    #     articles.append(article_to_record(article))

    # メインの条文を抽出(COLIEEコーパス形式)
    for article in root.findall(".//article"):
        articles.append(article_to_record(article))
    
    return articles

# 各条文要素をdictに変換する関数
def article_to_record(article_elem):
    # 条文番号抽出
    num = article_elem.get('num_ar')

    # 段落 + 文をすべて連結
    sentences = []
    for sent in article_elem.findall(".//display"):
        if sent.text:
            sentences.append(sent.text.strip())

    text = " ".join(sentences)

    return {
        "num": num,
        "text": text,
    }