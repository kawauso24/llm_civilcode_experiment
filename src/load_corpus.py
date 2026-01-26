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

    articles = []
    # メインの条文を抽出(COLIEEコーパス形式)
    for article in root.findall("Article"):
        articles.append(article_to_record(article))
    
    return articles

# 各条文要素をdictに変換する関数
def article_to_record(article_elem):
    # 条文番号抽出
    num = article_elem.get('num')

    # キャプションの抽出(存在する場合)
    caption_elem = article_elem.find("caption")
    caption = caption_elem.text.strip() if caption_elem is not None and caption_elem.text else ""

    # 条文テキスト抽出
    text_elem = article_elem.find("text")
    text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""

    return {
        "num": num,
        "text": caption + text,
    }