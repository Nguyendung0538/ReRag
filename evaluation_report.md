# Báo Cáo Đánh Giá Hệ Thống RAG & Text Diff (Evaluation Report)
- **Thời gian chạy:** 2026-06-24 19:37:57
- **Mô hình sử dụng:** `gemma4:e4b`
- **Tổng số ca kiểm thử:** 11

## 1. Kết Quả Tổng Hợp (Summary Metrics)

### Hiệu năng Hệ thống (Latency)
| Tác vụ | Thời gian trung bình (giây) |
| :--- | :--- |
| Nạp & Index tài liệu | 29.98 s |
| Tìm sự khác biệt (Text Diff) | 0.03 s |
| Truy vấn & Lập luận (RAG) | 1.08 s |

### Chất lượng Đánh giá (Quality Metrics)
| Thành phần | Precision | Recall | F1-Score | Citation Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Text Diff Engine** | 95.45%| 100.00% | 96.97% | N/A |
| **RAG Pipeline** | 96.97% | 95.45% | 95.15% | 100.00% |

---
## 2. Chi Tiết Từng Ca Kiểm Thử (Detailed Test Cases)

### Thư mục: `chuyen_nhung_co_phan`
- Số lượng thay đổi thực tế (Ground Truth): **3**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 3 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 7: Bổ sung yêu cầu công chứng đối với các sửa đổi, bổ sung của Hợp đồng. (Bản cũ: `bằng văn bản và có chữ ký của các bên.` -> Bản mới: `bằng văn bản, có chữ ký của các bên và phải được công chứng.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc bổ sung yêu cầu phải công chứng đối với các sửa đổi, bổ sung. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 11: Thay đổi cơ quan giải quyết tranh chấp từ MCAC sang VIAC. (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Trung tâm Trọng tài Quốc tế Việt Nam (VIAC)`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc thay đổi cơ quan giải quyết tranh chấp từ MCAC sang VIAC. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 12: Tăng số lượng bản hợp đồng được lập từ 3 lên 4 bản. (Bản cũ: `03 (ba) bản` -> Bản mới: `04 (bốn) bản`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc tăng số lượng bản hợp đồng được lập từ 3 lên 4. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 3 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 7: Bổ sung yêu cầu công chứng đối với các sửa đổi, bổ sung của Hợp đồng. (Bản cũ: `bằng văn bản và có chữ ký của các bên.` -> Bản mới: `bằng văn bản, có chữ ký của các bên và phải được công chứng.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc bổ sung yêu cầu 'phải được công chứng' vào Điều 7. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 11: Thay đổi cơ quan giải quyết tranh chấp từ MCAC sang VIAC. (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Trung tâm Trọng tài Quốc tế Việt Nam (VIAC)`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc thay đổi cơ quan giải quyết tranh chấp từ MCAC sang VIAC tại Điều 11. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 12: Tăng số lượng bản hợp đồng được lập từ 3 lên 4 bản. (Bản cũ: `03 (ba) bản` -> Bản mới: `04 (bốn) bản`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc tăng số lượng bản hợp đồng từ 03 lên 04 tại Điều 12. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `dai_ly`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 12: Thay đổi tổ chức trọng tài giải quyết tranh chấp (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Trung tâm Trọng tài Quốc tế Việt Nam (VIAC)`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi tổ chức trọng tài từ MCAC sang VIAC. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 12: Thay đổi địa điểm giải quyết tranh chấp (Bản cũ: `thành phố Đà Nẵng` -> Bản mới: `TP. Hồ Chí Minh`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi địa điểm giải quyết tranh chấp từ 'thành phố Đà Nẵng' sang 'TP. Hồ Chí Minh'. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 12: Thay đổi tổ chức trọng tài giải quyết tranh chấp (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Trung tâm Trọng tài Quốc tế Việt Nam (VIAC)`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi tổ chức trọng tài từ MCAC sang VIAC. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 12: Thay đổi địa điểm giải quyết tranh chấp (Bản cũ: `thành phố Đà Nẵng` -> Bản mới: `TP. Hồ Chí Minh`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi địa điểm giải quyết tranh chấp từ thành phố Đà Nẵng sang TP. Hồ Chí Minh. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `dich_vu_sua_chua`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 3: Thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% (Bản cũ: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 30 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.` -> Bản mới: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 50 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% tại Điều 3. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Tăng mức phạt chậm thanh toán từ 1% lên 2% (Bản cũ: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 1% (một phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.` -> Bản mới: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 2% (hai phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc tăng mức phạt chậm thanh toán từ 1% lên 2% tại Điều 9. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 3: Thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% (Bản cũ: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 30 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.` -> Bản mới: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 50 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% tại Điều 3. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Tăng mức phạt chậm thanh toán từ 1% lên 2% (Bản cũ: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 1% (một phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.` -> Bản mới: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 2% (hai phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc tăng mức phạt chậm thanh toán từ 1% lên 2% tại Điều 9. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `dich_vu_sua_chua!!`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 50.00% | **Recall:** 100.00% | **F1-Score:** 66.67%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 2 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 3: Thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% (Bản cũ: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 30 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.` -> Bản mới: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 50 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% tại Điều 3. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Tăng mức phạt chậm thanh toán từ 1% lên 2% (Bản cũ: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 1% (một phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.` -> Bản mới: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 2% (hai phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc tăng mức phạt chậm thanh toán từ 1% lên 2% tại Điều 9. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):**
    - **[Dư thừa]** Vị trí: `Điều 8` | Mô tả: Bổ sung nội dung không liên quan và có vẻ là lỗi đánh máy/thêm thừa ('ĐIỀU .9 SCAM lamlamo') vào cuối Điều 8 ở Bản mới.
    - **[Dư thừa]** Vị trí: `Điều 10, Điều 11, Điều 12` | Mô tả: Các thay đổi về cấu trúc điều khoản (số thứ tự, chuyển từ Điều 10 sang Điều 11, và các điều chỉnh sau đó) không được liệt kê trong Ground Truth Changes.

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 66.67% | **Recall:** 100.00% | **F1-Score:** 80.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 1 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 3: Thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% (Bản cũ: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 30 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.` -> Bản mới: `Bên A tạm ứng cho Bên B số tiền là ………. đồng (………..đồng chẵn) tương ứng 50 % giá trị Hợp đồng ngay sau khi ký bản hợp đồng này.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi tỷ lệ phần trăm tạm ứng từ 30% lên 50% tại Điều 3. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Tăng mức phạt chậm thanh toán từ 1% lên 2% (Bản cũ: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 1% (một phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.` -> Bản mới: `Nếu Bên A vi phạm điều khoản thanh toán thì Bên A sẽ chịu phạt 2% (hai phần trăm) của tổng số tiền phải thanh toán cho Bên B trên một ngày chậm thanh toán.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc tăng mức phạt chậm thanh toán từ 1% lên 2% tại Điều 9. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):**
    - **[Dư thừa]** Vị trí: `Điều 8` | Mô tả: Thêm nội dung không liên quan và sai lệch ('ĐIỀU .9 SCAM lamlamo') vào cuối phần 'Bản mới' của Điều 8.

---
### Thư mục: `Hop_tac_dau_tu`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 18: Bổ sung yêu cầu công chứng khi thông báo bằng văn bản (Bản cũ: `bằng văn bản` -> Bản mới: `bằng văn bản và phải được công chứng`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung yêu cầu công chứng tại Điều 18. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 19: Tăng số lượng bản hợp đồng từ 2 lên 4 (Bản cũ: `02 bản` -> Bản mới: `04 bản`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng sự thay đổi số lượng bản hợp đồng từ 02 lên 04 tại Điều 19. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 18: Bổ sung yêu cầu công chứng khi thông báo bằng văn bản (Bản cũ: `bằng văn bản` -> Bản mới: `bằng văn bản và phải được công chứng`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung yêu cầu công chứng tại Điều 18. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 19: Tăng số lượng bản hợp đồng từ 2 lên 4 (Bản cũ: `02 bản` -> Bản mới: `04 bản`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng sự thay đổi về số lượng bản hợp đồng từ 02 lên 04 tại Điều 19. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `hop_tac_kinh_doanh`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 6: Sửa đổi tiêu đề điều khoản thành quyền và trách nhiệm bồi thường (Bản cũ: `QUYỀN VÀ NGHĨA VỤ CỦA BÊN A` -> Bản mới: `QUYỀN VÀ TRÁCH NHIỆM BỒI THƯỜNG CỦA BÊN A`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi tiêu đề điều khoản từ 'Nghĩa vụ' sang 'Trách nhiệm Bồi thường' tại Điều 6. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 2: Xóa bỏ phần quy định về sử dụng tư cách hộ kinh doanh (Bản cũ: `Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác và sử dụng tư cách của Hộ kinh doanh ………….do Bên A sở hữu và các đầy đủ các Giấy phép kinh doanh để kinh doanh nhà hàng quán bar để kinh doanh Đơn vị hợp tác.` -> Bản mới: `Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc xóa bỏ phần quy định về sử dụng tư cách hộ kinh doanh và các giấy phép liên quan tại Điều 2. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 6: Sửa đổi tiêu đề điều khoản thành quyền và trách nhiệm bồi thường (Bản cũ: `QUYỀN VÀ NGHĨA VỤ CỦA BÊN A` -> Bản mới: `QUYỀN VÀ TRÁCH NHIỆM BỒI THƯỜNG CỦA BÊN A`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc thay đổi tiêu đề điều khoản từ 'QUYỀN VÀ NGHĨA VỤ' thành 'QUYỀN VÀ TRÁCH NHIỆM BỒI THƯỜNG' tại Điều 6. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 2: Xóa bỏ phần quy định về sử dụng tư cách hộ kinh doanh (Bản cũ: `Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác và sử dụng tư cách của Hộ kinh doanh ………….do Bên A sở hữu và các đầy đủ các Giấy phép kinh doanh để kinh doanh nhà hàng quán bar để kinh doanh Đơn vị hợp tác.` -> Bản mới: `Đối với Bên A: Bên A góp toàn bộ các tài sản tại Địa điểm hợp tác.`)
      *LLM-Judge:* Hệ thống đã nhận diện đúng việc loại bỏ phần quy định chi tiết về sử dụng tư cách hộ kinh doanh và giấy phép kinh doanh tại Điều 2. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `li_xang`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 12: Rút ngắn thời hạn từ 30 xuống 15 ngày (Bản cũ: `30 ngày` -> Bản mới: `15 ngày`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc rút ngắn thời hạn từ 30 ngày xuống 15 ngày tại Điều 12. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 13: Tăng số lượng bản hợp đồng từ 4 lên 6 (Bản cũ: `04 (bốn)` -> Bản mới: `06 (sáu)`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc tăng số lượng bản hợp đồng từ 04 (bốn) lên 06 (sáu) tại Điều 13. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 12: Rút ngắn thời hạn từ 30 xuống 15 ngày (Bản cũ: `30 ngày` -> Bản mới: `15 ngày`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi thời hạn từ 30 ngày xuống 15 ngày tại Điều 12. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 13: Tăng số lượng bản hợp đồng từ 4 lên 6 (Bản cũ: `04 (bốn)` -> Bản mới: `06 (sáu)`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi số lượng bản hợp đồng từ 04 (bốn) lên 06 (sáu) tại Điều 13. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `moi_gioi_mua_ban_bat_dong_san`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 5: Bổ sung thời hạn thanh toán phí môi giới là 05 ngày (Bản cũ: `- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng;` -> Bản mới: `- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng trong thời hạn 05 ngày;`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung cụm từ 'trong thời hạn 05 ngày' tại Điều 5. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 1: Bổ sung từ độc quyền vào dịch vụ môi giới (Bản cũ: `Bên B đồng ý giao cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.` -> Bản mới: `Bên B đồng ý giao độc quyền cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung từ 'độc quyền' vào nội dung tại Điều 1. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 5: Bổ sung thời hạn thanh toán phí môi giới là 05 ngày (Bản cũ: `- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng;` -> Bản mới: `- Thanh toán phí môi giới cho bên A theo Điều 2 của hợp đồng trong thời hạn 05 ngày;`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung thời hạn thanh toán phí môi giới là 05 ngày tại Điều 5. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 1: Bổ sung từ độc quyền vào dịch vụ môi giới (Bản cũ: `Bên B đồng ý giao cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.` -> Bản mới: `Bên B đồng ý giao độc quyền cho Bên A thực hiện dịch vụ môi giới bán/mua bất động sản do Bên B là chủ sở hữu.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung từ 'độc quyền' vào dịch vụ môi giới tại Điều 1. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `mua_ban_hang_hoa`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 13: Rút ngắn thời hạn từ 30 xuống 20 ngày (Bản cũ: `30 ngày` -> Bản mới: `20 ngày`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc rút ngắn thời hạn khắc phục vi phạm từ 30 ngày xuống còn 20 ngày tại Điều 13. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 14: Thay đổi phương thức giải quyết tranh chấp từ trọng tài sang tòa án (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Tòa án nhân dân có thẩm quyền`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi phương thức giải quyết tranh chấp từ Trung tâm Trọng tài Thương mại Miền Trung (MCAC) sang Tòa án nhân dân có thẩm quyền tại Điều 14. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 13: Rút ngắn thời hạn từ 30 xuống 20 ngày (Bản cũ: `30 ngày` -> Bản mới: `20 ngày`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi giá trị thời hạn từ 30 ngày xuống 20 ngày tại Điều 13. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 14: Thay đổi phương thức giải quyết tranh chấp từ trọng tài sang tòa án (Bản cũ: `Trung tâm Trọng tài Thương mại Miền Trung (MCAC)` -> Bản mới: `Tòa án nhân dân có thẩm quyền`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi phương thức giải quyết tranh chấp từ Trung tâm Trọng tài Thương mại Miền Trung (MCAC) sang Tòa án nhân dân có thẩm quyền tại Điều 14. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `nguyen_tac`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 2: Xóa cụm từ và các tài liệu khác liên quan (Bản cũ: `Trong phạm vi hợp đồng này và các tài liệu khác liên quan và gắn liền với Hợp đồng này, các từ ngữ dưới đây được hiểu như sau:` -> Bản mới: `Trong phạm vi hợp đồng này, các từ ngữ dưới đây được hiểu như sau:`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc xóa cụm từ 'và các tài liệu khác liên quan và gắn liền với Hợp đồng này' tại Điều 2. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Làm rõ pháp luật áp dụng là pháp luật sở hữu trí tuệ Việt Nam hiện hành (Bản cũ: `Các Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật.` -> Bản mới: `Các Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật sở hữu trí tuệ Việt Nam hiện hành.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung và làm rõ nội dung pháp luật áp dụng tại Điều 9 từ 'pháp luật' thành 'pháp luật sở hữu trí tuệ Việt Nam hiện hành'. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 2: Xóa cụm từ và các tài liệu khác liên quan (Bản cũ: `Trong phạm vi hợp đồng này và các tài liệu khác liên quan và gắn liền với Hợp đồng này, các từ ngữ dưới đây được hiểu như sau:` -> Bản mới: `Trong phạm vi hợp đồng này, các từ ngữ dưới đây được hiểu như sau:`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc xóa cụm từ 'và các tài liệu khác liên quan và gắn liền với Hợp đồng này' tại Điều 2. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 9: Làm rõ pháp luật áp dụng là pháp luật sở hữu trí tuệ Việt Nam hiện hành (Bản cũ: `Các Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật.` -> Bản mới: `Các Bên cam kết không vi phạm quyền sở hữu trí tuệ của nhau trong quá trình thực hiện dự án theo quy định của pháp luật sở hữu trí tuệ Việt Nam hiện hành.`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc bổ sung và làm rõ nội dung pháp luật áp dụng tại Điều 9 thành 'pháp luật sở hữu trí tuệ Việt Nam hiện hành'. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---
### Thư mục: `tham_dinh_gia`
- Số lượng thay đổi thực tế (Ground Truth): **2**

#### 🔹 Kết quả của Text Diff Engine
  - **Precision:** 100.00% | **Recall:** 100.00% | **F1-Score:** 100.00%
  - Thống kê thay đổi: **TP:** 2 | **FP:** 0 | **FN:** 0
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Lời nói đầu / Căn cứ: Thay đổi địa chỉ của Bên B (Bản cũ: `quận Hai Bà Trưng` -> Bản mới: `quận Cầu Giấy`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi địa chỉ từ 'quận Hai Bà Trưng' sang 'quận Cầu Giấy'. | Trích dẫn: ✅ Đúng
    - **[Khớp]** Điều 7: Cụ thể hóa thời hạn thông báo từ 'kịp thời' thành 03 ngày làm việc (Bản cũ: `kịp thời thông báo` -> Bản mới: `thông báo trong vòng 03 ngày làm việc`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc cụ thể hóa thời hạn thông báo từ 'kịp thời thông báo' thành 'thông báo trong vòng 03 ngày làm việc'. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

#### 🔸 Kết quả của RAG Pipeline
  - **Precision:** 100.00% | **Recall:** 50.00% | **F1-Score:** 66.67%
  - Thống kê thay đổi: **TP:** 1 | **FP:** 0 | **FN:** 1
  - **Độ chính xác trích dẫn (Citation Accuracy):** 100.00%
  - **✅ Các thay đổi bắt được đúng (True Positives):**
    - **[Khớp]** Điều 7: Cụ thể hóa thời hạn thông báo từ 'kịp thời' thành 03 ngày làm việc (Bản cũ: `kịp thời thông báo` -> Bản mới: `thông báo trong vòng 03 ngày làm việc`)
      *LLM-Judge:* Hệ thống đã nhận diện chính xác việc thay đổi cụ thể hóa thời hạn thông báo từ 'kịp thời' thành 'trong vòng 03 ngày làm việc' tại Điều 7. | Trích dẫn: ✅ Đúng
  - **❌ Các thay đổi bị bỏ sót (False Negatives):**
    - **[Bỏ sót]** Lời nói đầu / Căn cứ: Thay đổi địa chỉ của Bên B (Bản cũ: `quận Hai Bà Trưng` -> Bản mới: `quận Cầu Giấy`)
  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có

---