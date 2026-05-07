import docx

doc_path = r"d:\Code\ReRag\Test_data\chuyen_nhung_co_phan\v1_chuyen_nhung_co_phan.docx"
doc = docx.Document(doc_path)
text = []
for i, para in enumerate(doc.paragraphs):
    if para.text.strip():
        text.append(f"[{i}] {para.text}")

with open("chuyen_nhung_co_phan_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text))
