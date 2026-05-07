import docx
import os
import json

test_data_dir = r"d:\Code\ReRag\Test_data"
contracts = [
    "dich_vu_sua_chua", "giao_khoan", "hop_tac_kinh_doanh",
    "moi_gioi_mua_ban_bat_dong_san", "nguyen_tac", "phan_phoi", "quang_cao_thuong_mai",
    "tu_van_thiet_ke", "uy_thac_nhap_khau", "uy_thac_xuat_khau", "van_chuyen"
]

results = {}

for contract in contracts:
    doc_path = os.path.join(test_data_dir, contract, f"v1_{contract}.docx")
    if not os.path.exists(doc_path):
        continue
        
    doc = docx.Document(doc_path)
    paras = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]
    
    if len(paras) < 3:
        results[contract] = paras
        continue
        
    # Take 3 paragraphs: one near start (but skip titles), one middle, one near end
    idx1 = len(paras) // 4
    idx2 = len(paras) // 2
    idx3 = int(len(paras) * 0.8)
    
    results[contract] = [
        paras[idx1],
        paras[idx2],
        paras[idx3]
    ]

with open("samples.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
