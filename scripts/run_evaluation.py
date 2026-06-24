import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# python scripts/run_evaluation.py --limit 2 --model gemma4:e4b

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_processor import process_document
from src.indexing_strategies.tradi_rag import TradiRAGIndexing
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.diff.text_diff_engine import TextDiffEngine
from src.evaluation.metrics_evaluator import MetricsEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Hệ thống đánh giá tự động (Evaluation & Metrics Framework) cho ReRag.")
    parser.add_argument("--test-dir", type=str, default="Test_data", help="Thư mục chứa dữ liệu test.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng thư mục test cần chạy.")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Mô hình LLM được sử dụng cho RAG và Đánh giá.")
    parser.add_argument("--query", type=str, default="Hãy liệt kê chi tiết các điểm thay đổi, sửa đổi, bổ sung hoặc xóa bỏ giữa hai bản hợp đồng.", help="Câu hỏi truy vấn RAG.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    test_dir = args.test_dir
    if not os.path.exists(test_dir):
        print(f"❌ Thư mục test không tồn tại: {test_dir}")
        sys.exit(1)
        
    # Lấy danh sách các thư mục con chứa ground_truth.json
    subdirs = []
    for d in os.listdir(test_dir):
        subdir_path = os.path.join(test_dir, d)
        if os.path.isdir(subdir_path) and "ground_truth.json" in os.listdir(subdir_path):
            subdirs.append(subdir_path)
            
    if not subdirs:
        print(f"⚠ Không tìm thấy thư mục test nào chứa ground_truth.json trong {test_dir}")
        sys.exit(0)
        
    if args.limit:
        subdirs = subdirs[:args.limit]
        print(f"ℹ Chạy giới hạn {args.limit} thư mục test đầu tiên.")

    print(f"🚀 Bắt đầu đánh giá tự động trên {len(subdirs)} ca kiểm thử với model '{args.model}'...")
    
    llm_client = LLMClient(model_name=args.model)
    evaluator = MetricsEvaluator(llm_client=llm_client)
    
    results = []
    
    for i, subdir in enumerate(subdirs):
        folder_name = os.path.basename(subdir)
        print(f"\n[{i+1}/{len(subdirs)}] Đang xử lý: {folder_name}...")
        
        gt_path = os.path.join(subdir, "ground_truth.json")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        v1_path = os.path.join(subdir, gt_data["v1_file"])
        v2_path = os.path.join(subdir, gt_data["v2_file"])
        
        if not os.path.exists(v1_path) or not os.path.exists(v2_path):
            print(f"  ❌ Thiếu file docx nguồn ({gt_data['v1_file']} hoặc {gt_data['v2_file']}). Bỏ qua.")
            continue
            
        # 1. Ingestion & Indexing
        print("  📥 Bước 1: Nạp và Lập chỉ mục tài liệu...")
        start_ingest = time.time()
        
        # Reset DB và tạo Indexing Strategy
        indexer = TradiRAGIndexing()
        
        chunks_v1 = process_document(v1_path)
        for chunk in chunks_v1:
            chunk.metadata["source"] = os.path.basename(v1_path)
            
        chunks_v2 = process_document(v2_path)
        for chunk in chunks_v2:
            chunk.metadata["source"] = os.path.basename(v2_path)
            
        indexer.index(chunks_v1 + chunks_v2)
        ingest_time = time.time() - start_ingest
        print(f"  ✅ Hoàn tất nạp dữ liệu. Thời gian: {ingest_time:.2f} giây.")
        
        # 2. Text Diff Engine
        print("  🔍 Bước 2: Đánh giá Text Diff Engine...")
        results_old = indexer.get_all_by_source(os.path.basename(v1_path))
        results_new = indexer.get_all_by_source(os.path.basename(v2_path))
        
        metas_old = results_old.get("metadatas", [[]])[0] if results_old else []
        texts_old = results_old.get("documents", [[]])[0] if results_old else []
        metas_new = results_new.get("metadatas", [[]])[0] if results_new else []
        texts_new = results_new.get("documents", [[]])[0] if results_new else []
        
        diff_engine = TextDiffEngine()
        start_diff = time.time()
        diff_text = diff_engine.diff_paired_chunks(
            chunks_old=metas_old,
            texts_old=texts_old,
            chunks_new=metas_new,
            texts_new=texts_new
        )
        diff_time = time.time() - start_diff
        
        # Chạy evaluator đánh giá Text Diff
        diff_metrics = evaluator.evaluate_diff(diff_text, gt_data["changes"])
        print(f"  ✅ Đã đánh giá Text Diff. F1-Score: {diff_metrics['f1']:.2f}")
        print_terminal_details(diff_metrics, gt_data["changes"], "Text Diff Engine")
        
        # 3. RAG Pipeline
        print("  🤖 Bước 3: Đánh giá RAG Pipeline (Retrieval & Generation)...")
        rag_engine = LegalRAGEngine(
            indexing_strategy=indexer,
            llm_client=llm_client,
            old_law_source=os.path.basename(v1_path),
            new_law_source=os.path.basename(v2_path)
        )
        
        start_rag = time.time()
        rag_response = "".join(rag_engine.stream_ask(query=args.query, top_k=12))
        rag_time = time.time() - start_rag
        
        # Chạy evaluator đánh giá RAG Pipeline
        rag_metrics = evaluator.evaluate_rag(rag_response, gt_data["changes"])
        print(f"  ✅ Đã đánh giá RAG Pipeline. F1-Score: {rag_metrics['f1']:.2f}")
        print_terminal_details(rag_metrics, gt_data["changes"], "RAG Pipeline")
        
        case_res = {
            "folder": folder_name,
            "ingest_time_seconds": ingest_time,
            "diff_time_seconds": diff_time,
            "rag_time_seconds": rag_time,
            "diff_eval": diff_metrics,
            "rag_eval": rag_metrics,
            "ground_truth_changes_count": len(gt_data["changes"]),
            "ground_truth_changes": gt_data["changes"]
        }
        results.append(case_res)
        
    # Tính toán chỉ số trung bình (Averages)
    total_cases = len(results)
    if total_cases == 0:
        print("❌ Không có kết quả đánh giá nào được sinh ra.")
        return
        
    avg_ingest = sum(r["ingest_time_seconds"] for r in results) / total_cases
    avg_diff_time = sum(r["diff_time_seconds"] for r in results) / total_cases
    avg_rag_time = sum(r["rag_time_seconds"] for r in results) / total_cases
    
    avg_diff_precision = sum(r["diff_eval"]["precision"] for r in results) / total_cases
    avg_diff_recall = sum(r["diff_eval"]["recall"] for r in results) / total_cases
    avg_diff_f1 = sum(r["diff_eval"]["f1"] for r in results) / total_cases
    
    avg_rag_precision = sum(r["rag_eval"]["precision"] for r in results) / total_cases
    avg_rag_recall = sum(r["rag_eval"]["recall"] for r in results) / total_cases
    avg_rag_f1 = sum(r["rag_eval"]["f1"] for r in results) / total_cases
    avg_rag_citation = sum(r["rag_eval"]["citation_accuracy"] for r in results) / total_cases

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": args.model,
        "total_cases": total_cases,
        "averages": {
            "ingest_time_seconds": avg_ingest,
            "diff_time_seconds": avg_diff_time,
            "rag_time_seconds": avg_rag_time,
            "diff_engine": {
                "precision": avg_diff_precision,
                "recall": avg_diff_recall,
                "f1": avg_diff_f1
            },
            "rag_pipeline": {
                "precision": avg_rag_precision,
                "recall": avg_rag_recall,
                "f1": avg_rag_f1,
                "citation_accuracy": avg_rag_citation
            }
        },
        "cases": results
    }
    
    # Ghi file JSON kết quả chi tiết
    results_json_path = "evaluation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Đã lưu kết quả chi tiết vào: {results_json_path}")
    
    # Tạo báo cáo Markdown tuyệt đẹp
    report_path = "evaluation_report.md"
    create_markdown_report(report_path, summary)
    print(f"📑 Đã xuất báo cáo đánh giá dạng Markdown vào: {report_path}")

def print_terminal_details(eval_result: Dict[str, Any], ground_truth_changes: List[Dict[str, Any]], component_name: str):
    details = eval_result.get("details", {})
    matched_map = {item["gt_id"]: item for item in details.get("matched_changes", [])}
    
    print(f"    === Chi tiết đánh giá {component_name} ===")
    for gt in ground_truth_changes:
        gt_id = gt["id"]
        gt_desc = f"{gt.get('location', 'N/A')}: {gt.get('description', '')}"
        is_captured = False
        citation_correct = None
        if gt_id in matched_map:
            is_captured = matched_map[gt_id].get("is_captured", False)
            citation_correct = matched_map[gt_id].get("citation_correct", None)
        
        cit_str = ""
        if citation_correct is not None:
            cit_str = " | Trích dẫn: " + ("✅ Đúng" if citation_correct else "❌ Sai")
            
        if is_captured:
            print(f"      ✅ [Khớp] GT ID {gt_id} ({gt_desc}){cit_str}")
        else:
            print(f"      ❌ [Bỏ sót] GT ID {gt_id} ({gt_desc})")
            
    extra = details.get("extra_changes", [])
    if extra:
        print("      ⚠️ [Dư thừa / Báo sai]:")
        for ext in extra:
            print(f"        - {ext.get('location', 'N/A')}: {ext.get('description', '')}")

def format_detailed_results(eval_result: Dict[str, Any], ground_truth_changes: List[Dict[str, Any]]) -> List[str]:
    lines = []
    details = eval_result.get("details", {})
    matched_map = {item["gt_id"]: item for item in details.get("matched_changes", [])}
    
    captured = []
    missed = []
    
    for gt in ground_truth_changes:
        gt_id = gt["id"]
        gt_desc = f"{gt.get('location', 'N/A')}: {gt.get('description', '')} (Bản cũ: `{gt.get('v1_text', '')}` -> Bản mới: `{gt.get('v2_text', '')}`)"
        
        is_captured = False
        explanation = "N/A"
        citation_correct = None
        
        if gt_id in matched_map:
            is_captured = matched_map[gt_id].get("is_captured", False)
            explanation = matched_map[gt_id].get("explanation", "N/A")
            citation_correct = matched_map[gt_id].get("citation_correct", None)
            
        if is_captured:
            citation_status = ""
            if citation_correct is not None:
                citation_status = " | Trích dẫn: " + ("✅ Đúng" if citation_correct else "❌ Sai")
            captured.append(f"    - **[Khớp]** {gt_desc}\n      *LLM-Judge:* {explanation}{citation_status}")
        else:
            missed.append(f"    - **[Bỏ sót]** {gt_desc}")
            
    extra = []
    for ext in details.get("extra_changes", []):
        extra.append(f"    - **[Dư thừa]** Vị trí: `{ext.get('location', 'N/A')}` | Mô tả: {ext.get('description', '')}")
        
    if captured:
        lines.append("  - **✅ Các thay đổi bắt được đúng (True Positives):**")
        lines.extend(captured)
    else:
        lines.append("  - **✅ Các thay đổi bắt được đúng (True Positives):** Không có")
        
    if missed:
        lines.append("  - **❌ Các thay đổi bị bỏ sót (False Negatives):**")
        lines.extend(missed)
    else:
        lines.append("  - **❌ Các thay đổi bị bỏ sót (False Negatives):** Không có")
        
    if extra:
        lines.append("  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):**")
        lines.extend(extra)
    else:
        lines.append("  - **⚠️ Các thay đổi dư thừa/báo cáo sai (False Positives):** Không có")
        
    return lines

def create_markdown_report(filepath: str, summary: Dict[str, Any]):
    lines = [
        "# Báo Cáo Đánh Giá Hệ Thống RAG & Text Diff (Evaluation Report)",
        f"- **Thời gian chạy:** {summary['timestamp']}",
        f"- **Mô hình sử dụng:** `{summary['model_used']}`",
        f"- **Tổng số ca kiểm thử:** {summary['total_cases']}",
        "",
        "## 1. Kết Quả Tổng Hợp (Summary Metrics)",
        "",
        "### Hiệu năng Hệ thống (Latency)",
        "| Tác vụ | Thời gian trung bình (giây) |",
        "| :--- | :--- |",
        f"| Nạp & Index tài liệu | {summary['averages']['ingest_time_seconds']:.2f} s |",
        f"| Tìm sự khác biệt (Text Diff) | {summary['averages']['diff_time_seconds']:.2f} s |",
        f"| Truy vấn & Lập luận (RAG) | {summary['averages']['rag_time_seconds']:.2f} s |",
        "",
        "### Chất lượng Đánh giá (Quality Metrics)",
        "| Thành phần | Precision | Recall | F1-Score | Citation Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Text Diff Engine** | {summary['averages']['diff_engine']['precision']:.2%}| {summary['averages']['diff_engine']['recall']:.2%} | {summary['averages']['diff_engine']['f1']:.2%} | N/A |",
        f"| **RAG Pipeline** | {summary['averages']['rag_pipeline']['precision']:.2%} | {summary['averages']['rag_pipeline']['recall']:.2%} | {summary['averages']['rag_pipeline']['f1']:.2%} | {summary['averages']['rag_pipeline']['citation_accuracy']:.2%} |",
        "",
        "---",
        "## 2. Chi Tiết Từng Ca Kiểm Thử (Detailed Test Cases)",
        ""
    ]
    
    for case in summary["cases"]:
        lines.extend([
            f"### Thư mục: `{case['folder']}`",
            f"- Số lượng thay đổi thực tế (Ground Truth): **{case['ground_truth_changes_count']}**",
            "",
            "#### 🔹 Kết quả của Text Diff Engine",
            f"  - **Precision:** {case['diff_eval']['precision']:.2%} | **Recall:** {case['diff_eval']['recall']:.2%} | **F1-Score:** {case['diff_eval']['f1']:.2%}",
            f"  - Thống kê thay đổi: **TP:** {case['diff_eval']['tp']} | **FP:** {case['diff_eval']['fp']} | **FN:** {case['diff_eval']['fn']}",
        ])
        lines.extend(format_detailed_results(case["diff_eval"], case["ground_truth_changes"]))
        
        lines.extend([
            "",
            "#### 🔸 Kết quả của RAG Pipeline",
            f"  - **Precision:** {case['rag_eval']['precision']:.2%} | **Recall:** {case['rag_eval']['recall']:.2%} | **F1-Score:** {case['rag_eval']['f1']:.2%}",
            f"  - Thống kê thay đổi: **TP:** {case['rag_eval']['tp']} | **FP:** {case['rag_eval']['fp']} | **FN:** {case['rag_eval']['fn']}",
            f"  - **Độ chính xác trích dẫn (Citation Accuracy):** {case['rag_eval']['citation_accuracy']:.2%}",
        ])
        lines.extend(format_detailed_results(case["rag_eval"], case["ground_truth_changes"]))
        lines.append("\n---")
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
