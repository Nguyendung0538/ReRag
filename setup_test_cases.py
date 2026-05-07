import os
import shutil
import json

base_doc_dir = r"d:\Code\ReRag\document\Hop_Dong"
test_data_dir = r"d:\Code\ReRag\Test_data"

contracts = [
    "chuyen_nhung_co_phan", "dich_vu_sua_chua", "giao_khoan", "hop_tac_kinh_doanh",
    "moi_gioi_mua_ban_bat_dong_san", "nguyen_tac", "phan_phoi", "quang_cao_thuong_mai",
    "tu_van_thiet_ke", "uy_thac_nhap_khau", "uy_thac_xuat_khau", "van_chuyen"
]

for idx, contract in enumerate(contracts):
    # pair_id start from 02 (assuming 01 is hop_tac_dau_tu)
    pair_id = f"pair_{str(idx+2).zfill(2)}"
    
    # Create directory
    target_dir = os.path.join(test_data_dir, contract)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files
    src_file = os.path.join(base_doc_dir, f"{contract}.docx")
    v1_file = os.path.join(target_dir, f"v1_{contract}.docx")
    v2_file = os.path.join(target_dir, f"v2_{contract}.docx")
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, v1_file)
        shutil.copy2(src_file, v2_file)
        
    # Generate ground_truth.json
    gt_data = {
        "pair_id": pair_id,
        "v1_file": f"v1_{contract}.docx",
        "v2_file": f"v2_{contract}.docx",
        "changes": [
            {
                "id": "c1",
                "type": "modification",
                "location": "Điều khoản chung",
                "v1_text": "bằng văn bản",
                "v2_text": "bằng văn bản và phải được công chứng",
                "description": "Bổ sung yêu cầu công chứng khi thông báo bằng văn bản"
            },
            {
                "id": "c2",
                "type": "modification",
                "location": "Giải quyết tranh chấp",
                "v1_text": "Tòa án",
                "v2_text": "Trung tâm Trọng tài Quốc tế Việt Nam (VIAC)",
                "description": "Thay đổi cơ quan giải quyết tranh chấp từ Tòa án sang Trọng tài"
            }
        ]
    }
    
    gt_file = os.path.join(target_dir, "ground_truth.json")
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
        
print(f"Created folders and files for {len(contracts)} contracts.")
