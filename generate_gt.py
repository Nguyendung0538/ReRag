import os
import json

test_data_dir = r"d:\Code\ReRag\Test_data"

data = {
    "dich_vu_sua_chua": [
        {
            "id": "c1", "type": "value_change", "location": "Nội dung hợp đồng",
            "v1_text": "Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng…… % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.",
            "v2_text": "Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 50% giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.",
            "description": "Điền tỷ lệ phần trăm tạm ứng là 50%"
        },
        {
            "id": "c2", "type": "value_change", "location": "Nội dung hợp đồng",
            "v1_text": "- Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 1% (một phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.",
            "v2_text": "- Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 2% (hai phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.",
            "description": "Tăng mức phạt chậm thanh toán từ 1% lên 2%"
        }
    ],
    "giao_khoan": [
        {
            "id": "c1", "type": "modification", "location": "Quyền và nghĩa vụ",
            "v1_text": "6.2.  Quyền và nghĩa vụ của Bên B:",
            "v2_text": "6.2.  Quyền, nghĩa vụ và trách nhiệm của Bên B:",
            "description": "Bổ sung thêm chữ trách nhiệm vào tiêu đề điều khoản"
        },
        {
            "id": "c2", "type": "addition", "location": "Chấm dứt hợp đồng",
            "v1_text": "11.1.  Hợp đồng này có thể bị chấm dứt trước thời hạn theo một trong các trường hợp sau:.",
            "v2_text": "11.1.  Hợp đồng này có thể bị chấm dứt trước thời hạn theo một trong các trường hợp sau đây và phải báo trước 30 ngày:.",
            "description": "Bổ sung thời hạn báo trước 30 ngày khi chấm dứt hợp đồng"
        }
    ],
    "hop_tac_kinh_doanh": [
        {
            "id": "c1", "type": "modification", "location": "Quyền và nghĩa vụ",
            "v1_text": "ĐIỀU 6.  QUYỀN VÀ NGHĨA VỤ CỦA BÊN A",
            "v2_text": "ĐIỀU 6.  QUYỀN VÀ TRÁCH NHIỆM BỒI THƯỜNG CỦA BÊN A",
            "description": "Sửa đổi tiêu đề điều khoản thành quyền và trách nhiệm bồi thường"
        },
        {
            "id": "c2", "type": "deletion", "location": "Nội dung hợp tác",
            "v1_text": "Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác và sử dụng tư cách của Hộ kinh doanh ………….do Bên A sở hữu và các đầy đủ các Giấy phép kinh doanh để kinh doanh nhà hàng quán bar để kinh doanh Đơn vị hợp tác.",
            "v2_text": "Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác.",
            "description": "Xóa bỏ phần quy định về sử dụng tư cách hộ kinh doanh"
        }
    ],
    "moi_gioi_mua_ban_bat_dong_san": [
        {
            "id": "c1", "type": "addition", "location": "Nghĩa vụ",
            "v1_text": "- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng;",
            "v2_text": "- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng trong thời hạn 05 ngày;",
            "description": "Bổ sung thời hạn thanh toán phí môi giới là 05 ngày"
        },
        {
            "id": "c2", "type": "modification", "location": "Đối tượng hợp đồng",
            "v1_text": "1.1. Bên B đồng ý giao cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.",
            "v2_text": "1.1. Bên B đồng ý giao độc quyền cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.",
            "description": "Bổ sung từ độc quyền vào dịch vụ môi giới"
        }
    ],
    "nguyen_tac": [
        {
            "id": "c1", "type": "deletion", "location": "Giải thích từ ngữ",
            "v1_text": "Trong phạm vi hợp đồng này và các tài liệu khác liên quan và gắn liền với Hợp đồng này, các từ ngữ dưới đây được hiểu như sau:",
            "v2_text": "Trong phạm vi hợp đồng này, các từ ngữ dưới đây được hiểu như sau:",
            "description": "Xóa cụm từ và các tài liệu khác liên quan"
        },
        {
            "id": "c2", "type": "addition", "location": "Sở hữu trí tuệ",
            "v1_text": "9.2.\tCác Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật.",
            "v2_text": "9.2.\tCác Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật sở hữu trí tuệ Việt Nam hiện hành.",
            "description": "Làm rõ pháp luật áp dụng là pháp luật sở hữu trí tuệ Việt Nam hiện hành"
        }
    ],
    "phan_phoi": [
        {
            "id": "c1", "type": "modification", "location": "Chi phí",
            "v1_text": "Chi phí vận chuyển từ Công Ty đến địa điểm giao hàng do Công Ty chịu. Nhà Phân Phối chịu trách nhiệm tổ chức cho người để bốc dỡ hàng và chịu chi phó dỡ hàng.",
            "v2_text": "Chi phí vận chuyển từ Công Ty đến địa điểm giao hàng do Nhà Phân Phối chịu. Nhà Phân Phối chịu trách nhiệm tổ chức cho người để bốc dỡ hàng và chịu chi phó dỡ hàng.",
            "description": "Thay đổi bên chịu chi phí vận chuyển từ Công Ty sang Nhà Phân Phối"
        },
        {
            "id": "c2", "type": "deletion", "location": "Bất khả kháng",
            "v1_text": "tiến hành các biện pháp ngăn ngừa hợp lý và các biện pháp cần thiết để hạn chế tối đa các ảnh hưởng do Sự Kiện Bất Khả Kháng gây ra; và",
            "v2_text": "tiến hành các biện pháp ngăn ngừa hợp lý để hạn chế tối đa các ảnh hưởng do Sự Kiện Bất Khả Kháng gây ra; và",
            "description": "Lược bỏ cụm từ và các biện pháp cần thiết"
        }
    ],
    "quang_cao_thuong_mai": [
        {
            "id": "c1", "type": "modification", "location": "Thanh toán",
            "v1_text": "2 – Bên A thanh toán cho bên B bằng đồng Việt Nam bằng hình thức ………………………và được chia thành các đợt thanh toán sau:",
            "v2_text": "2 – Bên A thanh toán cho bên B bằng đồng Việt Nam bằng hình thức chuyển khoản và được chia thành các đợt thanh toán sau:",
            "description": "Cụ thể hóa hình thức thanh toán là chuyển khoản"
        },
        {
            "id": "c2", "type": "deletion", "location": "Thông báo",
            "v1_text": "Trong một số trường hợp đặc biệt khẩn cấp, một trong Hai Bên đồng ý hình thức thông báo cho Bên kia bằng điện thoại trực tiếp.",
            "v2_text": "Trong trường hợp khẩn cấp, một trong Hai Bên đồng ý hình thức thông báo bằng điện thoại trực tiếp.",
            "description": "Rút gọn câu thông báo khẩn cấp"
        }
    ],
    "tu_van_thiet_ke": [
        {
            "id": "c1", "type": "addition", "location": "Yêu cầu",
            "v1_text": "ĐIỀU 2. YÊU CẦU KỸ THUẬT, CHẤT LƯỢNG",
            "v2_text": "ĐIỀU 2. YÊU CẦU KỸ THUẬT, CHẤT LƯỢNG VÀ TIẾN ĐỘ",
            "description": "Bổ sung Tiến độ vào tiêu đề Điều 2"
        },
        {
            "id": "c2", "type": "modification", "location": "Bất khả kháng",
            "v1_text": "Trong trường hợp xảy ra sự kiện bất khả kháng, thời gian thực hiện hợp đồng sẽ được kéo dài bằng thời gian diễn ra sự kiện bất khả kháng mà bên bị ảnh hưởng không thể thực hiện các nghĩa vụ theo hợp đồng của mình.",
            "v2_text": "Trong trường hợp xảy ra sự kiện bất khả kháng, thời gian thực hiện hợp đồng sẽ được kéo dài tối đa 30 ngày.",
            "description": "Giới hạn thời gian kéo dài hợp đồng tối đa 30 ngày thay vì bằng thời gian diễn ra sự kiện"
        }
    ],
    "uy_thac_nhap_khau": [
        {
            "id": "c1", "type": "addition", "location": "Thanh toán",
            "v1_text": "b) Tổng cộng toàn bộ chi phí ủy thác mà bên A có trách nhiệm phài thanh toán cho bên B là:...",
            "v2_text": "b) Tổng cộng toàn bộ chi phí ủy thác đã bao gồm VAT mà bên A có trách nhiệm phài thanh toán cho bên B là:...",
            "description": "Bổ sung đã bao gồm VAT"
        },
        {
            "id": "c2", "type": "modification", "location": "Luật áp dụng",
            "v1_text": "Luật áp dụng cho hợp đồng này là… (chỉ áp dụng đối với tranh chấp có yếu tố nước ngoài và trong trường hợp các bên không thỏa thuận về luật áp dụng trong một điều khoản khác).",
            "v2_text": "Luật áp dụng cho hợp đồng này là Luật Thương mại Việt Nam.",
            "description": "Cụ thể hóa luật áp dụng là Luật Thương mại Việt Nam"
        }
    ],
    "uy_thac_xuat_khau": [
        {
            "id": "c1", "type": "modification", "location": "Thanh toán",
            "v1_text": "Điều 5: Thanh toán tiền bán hàng",
            "v2_text": "Điều 5: Phương thức và thời hạn thanh toán tiền bán hàng",
            "description": "Sửa tiêu đề điều khoản thanh toán rõ ràng hơn"
        },
        {
            "id": "c2", "type": "addition", "location": "Nội dung",
            "v1_text": "Bên A uỷ thác cho bên B xuất khẩu những mặt hàng sau:",
            "v2_text": "Bên A uỷ thác cho bên B xuất khẩu độc quyền những mặt hàng sau:",
            "description": "Bổ sung chữ độc quyền"
        }
    ],
    "van_chuyen": [
        {
            "id": "c1", "type": "deletion", "location": "Thanh toán",
            "v1_text": "Điều 3: Tạm ứng và Phương thức thanh toán",
            "v2_text": "Điều 3: Phương thức thanh toán",
            "description": "Xóa chữ Tạm ứng và ở tiêu đề"
        },
        {
            "id": "c2", "type": "value_change", "location": "Bồi thường",
            "v1_text": "Toàn bộ số tiền này phải được thanh toán dứt điểm cho bên A trong vòng 03 ngày kể từ ngày nhận được yêu cầu của bên A.",
            "v2_text": "Toàn bộ số tiền này phải được thanh toán dứt điểm cho bên A trong vòng 07 ngày kể từ ngày nhận được yêu cầu của bên A.",
            "description": "Tăng số ngày thanh toán từ 03 lên 07 ngày"
        }
    ]
}

idx = 3 # Starting after hop_tac_dau_tu (01) and chuyen_nhung_co_phan (02)
for contract, changes in data.items():
    pair_id = f"pair_{str(idx).zfill(2)}"
    idx += 1
    
    gt_data = {
        "pair_id": pair_id,
        "v1_file": f"v1_{contract}.docx",
        "v2_file": f"v2_{contract}.docx",
        "changes": changes
    }
    
    gt_file = os.path.join(test_data_dir, contract, "ground_truth.json")
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
        
print("Updated ground_truth.json for 11 remaining contracts.")
