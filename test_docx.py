import docx
import os

def replace_text_in_docx(doc_path, old_text, new_text, output_path):
    doc = docx.Document(doc_path)
    for paragraph in doc.paragraphs:
        if old_text in paragraph.text:
            paragraph.text = paragraph.text.replace(old_text, new_text)
            print(f"Replaced in paragraph: {paragraph.text[:50]}...")
            break
    doc.save(output_path)

input_file = r"d:\Code\ReRag\document\Hop_Dong\chuyen_nhung_co_phan.docx"
output_file = "test_replace.docx"
replace_text_in_docx(input_file, "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM (TEST)", output_file)
print(os.path.exists(output_file))
