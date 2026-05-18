import re
import difflib
from typing import List, Tuple

def _render_new_with_bold(new_tokens: List[str], opcodes: List[Tuple]) -> str:
    parts = []
    for tag, i1, i2, j1, j2 in opcodes:
        segment = " ".join(new_tokens[j1:j2])
        if tag == "equal":
            parts.append(segment)
        elif tag in ("replace", "insert"):
            if segment:
                parts.append(f"**{segment}**")
    return " ".join(parts)

def diff_texts_abbreviated(old_text: str, new_text: str) -> str:
    old_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', old_text) if s.strip()]
    new_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', new_text) if s.strip()]
    
    matcher = difflib.SequenceMatcher(None, old_sentences, new_sentences, autojunk=False)
    
    old_out = []
    new_out = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if not old_out or old_out[-1] != "[...]":
                old_out.append("[...]")
                new_out.append("[...]")
        else:
            sub_old_text = " ".join(old_sentences[i1:i2])
            sub_new_text = " ".join(new_sentences[j1:j2])
            
            old_tokens = sub_old_text.split()
            new_tokens = sub_new_text.split()
            sub_matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
            
            new_rendered = _render_new_with_bold(new_tokens, sub_matcher.get_opcodes())
            
            if sub_old_text:
                old_out.append(sub_old_text)
            if new_rendered:
                new_out.append(new_rendered)
                
    return f"  + Bản cũ: {' '.join(old_out)}\n  + Bản mới: {' '.join(new_out)}"

old = "1. Thời hạn thuê đã hết. 2. Nhà ở không còn. Nếu Bên A muốn chấm dứt hợp đồng thì phải báo trước."
new = "1. Thời hạn thuê đã hết. 2. Nhà ở không còn. Nếu Bên A muốn chấm dứt hợp đồng thì phải báo trước 30 ngày."

print(diff_texts_abbreviated(old, new))
