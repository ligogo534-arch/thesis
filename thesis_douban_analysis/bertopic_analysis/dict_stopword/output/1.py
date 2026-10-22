"""
ユーザー辞書（jieba形式: 単語 品詞 頻度）→ 単語だけを抽出してカンマ（読点）区切りに変換
"""
'''
#1.用户词典换成逗号分隔
input_path = "custom_dict_template.txt"       # 元のjiebaユーザー辞書パス
output_path = "dict_csv.txt"   # 出力先

words = []
with open(input_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # jieba辞書は空白（半角/全角どちらもあり得る）区切り。最初のトークンだけ単語
        word = line.split()[0]
        words.append(word)

comma_separated = ", ".join(words)  # 全角読点。半角カンマにしたい場合は "," に変更

with open(output_path, "w", encoding="utf-8") as f:
    f.write(comma_separated)

print(f"{len(words)}語を抽出・変換しました。")
print(comma_separated[:200] + "...")
'''

#2.新增停用词表换成逗号分隔
"""
停用词表（追加/削除リスト）: 1行1語のtxt → カンマ区切りのテキストに変換
"""

input_path = "stopwords_candidate.txt"      # 元のtxtファイルパス
output_path = "stopwords_csv.txt"  # 出力先

with open(input_path, encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]

comma_separated = ", ".join(words)  # 日本語論文なら全角読点「、」推奨。半角カンマなら "," に変更

with open(output_path, "w", encoding="utf-8") as f:
    f.write(comma_separated)

print(f"{len(words)}語を変換しました。")
print(comma_separated[:200] + "...")  # 先頭200文字だけプレビュー