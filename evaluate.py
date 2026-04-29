"""
Evaluation Pipeline — Đo lường chất lượng hệ thống RAG so sánh hợp đồng pháp lý.

Chạy: python evaluate.py
Yêu cầu: Ollama đang chạy với model gemma4:e4b + qwen3-embedding:8b

2 Tầng đánh giá:
  Tầng 1 (Diff): Đo ClauseDiffer phát hiện đúng bao nhiêu Điều có thay đổi (không cần LLM)
  Tầng 2 (RAG):  Đo toàn luồng RAG+LLM trả lời đúng bao nhiêu thay đổi (cần LLM)
"""

import json
import os
import re
import time
from typing import List, Dict, Any

from src.ingestion.document_processor import process_document
from src.diff.clause_differ import ClauseDiffer, ClauseDiff
from src.indexing_strategies.tradi_rag import TradiRAGIndexing
from src.generation.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine

# ── Cấu hình ────────────────────────────────────────────────────────────────
TEST_DATA_DIR = "Test_data"
LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "qwen3-embedding:8b"
BROAD_QUERY = "Liệt kê tất cả điểm khác biệt giữa 2 hợp đồng"


# ── Tiện ích ─────────────────────────────────────────────────────────────────

def load_ground_truth(test_dir: str) -> dict:
    """Đọc ground_truth.json từ thư mục test."""
    gt_path = os.path.join(test_dir, "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gt_dieu_set(ground_truth: dict) -> set:
    """
    Trích xuất tập hợp tên Điều từ ground truth.
    VD: {"Điều 12.4", "Điều 13.4"} → {"Điều 12", "Điều 13"} (cấp Điều cha)
    Và giữ cả tên gốc: {"Điều 12", "Điều 12.4", "Điều 13", "Điều 13.4"}
    """
    dieu_set = set()
    for change in ground_truth.get("changes", []):
        location = change.get("location", "")
        dieu_set.add(location)
        # Thêm Điều cha (VD: "Điều 12.4" → "Điều 12")
        parent_match = re.match(r"(Điều\s+\d+)", location)
        if parent_match:
            dieu_set.add(parent_match.group(1))
    return dieu_set


def parse_test_pair(test_dir: str, ground_truth: dict):
    """Parse 2 file DOCX thành chunks."""
    v1_path = os.path.join(test_dir, ground_truth["v1_file"])
    v2_path = os.path.join(test_dir, ground_truth["v2_file"])

    old_chunks = process_document(v1_path)
    for chunk in old_chunks:
        chunk.metadata["source"] = ground_truth["v1_file"]

    new_chunks = process_document(v2_path)
    for chunk in new_chunks:
        chunk.metadata["source"] = ground_truth["v2_file"]

    return old_chunks, new_chunks


# ── Tầng 1: Diff Evaluation ─────────────────────────────────────────────────

def evaluate_diff(test_dir: str, ground_truth: dict) -> dict:
    """
    Chạy ClauseDiffer trên 1 cặp file, so với ground_truth.
    Đo: ClauseDiffer có tìm đúng Điều chứa thay đổi không?
    
    Return: {precision, recall, f1, found_dieu, expected_dieu, details}
    """
    old_chunks, new_chunks = parse_test_pair(test_dir, ground_truth)

    differ = ClauseDiffer()
    diffs = differ.compare(old_chunks, new_chunks)

    # Tập Điều mà diff phát hiện
    diff_dieu_set = set()
    for d in diffs:
        diff_dieu_set.add(d.dieu)
        # Thêm Điều cha
        parent_match = re.match(r"(Điều\s+\d+)", d.dieu)
        if parent_match:
            diff_dieu_set.add(parent_match.group(1))

    # Tập Điều từ ground truth
    gt_dieu_set = extract_gt_dieu_set(ground_truth)

    # Tính metrics
    true_positives = diff_dieu_set & gt_dieu_set
    precision = len(true_positives) / len(diff_dieu_set) if diff_dieu_set else 0.0
    recall = len(true_positives) / len(gt_dieu_set) if gt_dieu_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "found_dieu": sorted(diff_dieu_set),
        "expected_dieu": sorted(gt_dieu_set),
        "true_positives": sorted(true_positives),
        "num_diffs": len(diffs),
    }


# ── Tầng 2: RAG Evaluation ──────────────────────────────────────────────────

def evaluate_rag(test_dir: str, ground_truth: dict) -> dict:
    """
    Chạy full RAG pipeline trên 1 cặp file.
    Hỏi broad query → kiểm tra output có chứa v1_text/v2_text từ GT không.
    
    Return: {change_recall, citation_hits, grounding_score, latency, details}
    """
    old_chunks, new_chunks = parse_test_pair(test_dir, ground_truth)
    all_chunks = old_chunks + new_chunks

    # Init RAG pipeline
    indexer = TradiRAGIndexing(embedding_model=EMBEDDING_MODEL)
    indexer.index(all_chunks)

    differ = ClauseDiffer()
    clause_diffs = differ.compare(old_chunks, new_chunks)

    llm_client = LLMClient(model_name=LLM_MODEL)
    engine = LegalRAGEngine(
        indexing_strategy=indexer,
        llm_client=llm_client,
        old_law_source=ground_truth["v1_file"],
        new_law_source=ground_truth["v2_file"],
        clause_diffs=clause_diffs,
    )

    # Chạy query
    start_time = time.time()
    answer = ""
    for chunk in engine.stream_ask(query=BROAD_QUERY, strategy_name="Normal_v1 (Raw Query)", top_k=6):
        answer += chunk
    latency = time.time() - start_time

    # Grounding score
    grounding = engine.compute_grounding_score(answer)

    # Xóa ký tự markdown in đậm/nghiêng để so sánh substring không bị fail
    answer_clean = re.sub(r'[*_]', '', answer.lower())

    # Kiểm tra từng change trong GT có xuất hiện trong output không
    changes = ground_truth.get("changes", [])
    hits = []
    for change in changes:
        v1_found = change["v1_text"].lower() in answer_clean
        v2_found = change["v2_text"].lower() in answer_clean
        hits.append({
            "id": change["id"],
            "location": change["location"],
            "v1_text": change["v1_text"],
            "v2_text": change["v2_text"],
            "v1_found": v1_found,
            "v2_found": v2_found,
            "both_found": v1_found and v2_found,
        })

    total_changes = len(changes)
    changes_detected = sum(1 for h in hits if h["both_found"])
    change_recall = (changes_detected / total_changes * 100) if total_changes > 0 else 0.0

    return {
        "change_recall": change_recall,
        "changes_detected": changes_detected,
        "total_changes": total_changes,
        "grounding_score": grounding,
        "latency": latency,
        "details": hits,
        "raw_answer": answer,
    }


# ── Report ───────────────────────────────────────────────────────────────────

def print_separator(char="═", width=90):
    print(char * width)


def print_report(all_results: List[Dict[str, Any]]):
    """In bảng tổng hợp kết quả."""
    print()
    print_separator()
    print(f"{'EVALUATION RESULTS':^90}")
    print_separator()

    # Header
    header = f"{'Test Pair':<22}│{'Diff P/R/F1':^18}│{'RAG Recall':^16}│{'Ground%':^10}│{'Time':^8}"
    print(header)
    print("─" * 22 + "┼" + "─" * 18 + "┼" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 8)

    # Data rows
    sum_diff_f1 = 0
    sum_rag_recall = 0
    sum_grounding = 0
    sum_latency = 0
    count = len(all_results)

    for r in all_results:
        name = r["name"][:20]
        dp = r["diff"]["precision"]
        dr = r["diff"]["recall"]
        df = r["diff"]["f1"]
        rr = r["rag"]["change_recall"]
        rd = r["rag"]["changes_detected"]
        rt = r["rag"]["total_changes"]
        gs = r["rag"]["grounding_score"]
        lt = r["rag"]["latency"]

        sum_diff_f1 += df
        sum_rag_recall += rr
        sum_grounding += gs
        sum_latency += lt

        diff_str = f"{dp:.0f}/{dr:.0f}/{df:.0f}"
        rag_str = f"{rd}/{rt} ({rr:.0f}%)"
        print(f" {name:<21}│ {diff_str:^16} │ {rag_str:^14} │ {gs:>6.0f}%  │ {lt:>5.1f}s")

    # Average
    print("─" * 22 + "┼" + "─" * 18 + "┼" + "─" * 16 + "┼" + "─" * 10 + "┼" + "─" * 8)
    avg_f1 = sum_diff_f1 / count if count else 0
    avg_recall = sum_rag_recall / count if count else 0
    avg_ground = sum_grounding / count if count else 0
    avg_latency = sum_latency / count if count else 0
    print(f" {'AVERAGE':<21}│ {'F1: %.0f' % avg_f1:^16} │ {'%.0f%%' % avg_recall:^14} │ {avg_ground:>6.0f}%  │ {avg_latency:>5.1f}s")
    print_separator()

    # Detailed per-test results
    print()
    print("=" * 90)
    print(f"{'CHI TIẾT TỪNG CẶP TEST':^90}")
    print("=" * 90)

    for r in all_results:
        print(f"\n{'─' * 90}")
        print(f"  {r['name']}")
        print(f"{'─' * 90}")

        # Diff details
        diff = r["diff"]
        print(f"  [Diff] Điều phát hiện: {', '.join(diff['found_dieu'])}")
        print(f"  [Diff] Điều expected:  {', '.join(diff['expected_dieu'])}")
        print(f"  [Diff] True positives: {', '.join(diff['true_positives'])}")
        print(f"  [Diff] P={diff['precision']:.0f}% R={diff['recall']:.0f}% F1={diff['f1']:.0f}%")

        # RAG details
        rag = r["rag"]
        print(f"  [RAG]  Grounding: {rag['grounding_score']:.0f}% | Latency: {rag['latency']:.1f}s")
        for hit in rag["details"]:
            v1_icon = "✅" if hit["v1_found"] else "❌"
            v2_icon = "✅" if hit["v2_found"] else "❌"
            status = 'PASS' if hit['both_found'] else 'FAIL'
            print(f"         {hit['location']}: v1={v1_icon} v2={v2_icon} → {status}")
            
            # Nếu FAIL, in ra câu trả lời gốc để debug
            if not hit['both_found']:
                print(f"            > Lỗi tìm kiếm substring:")
                if not hit["v1_found"]:
                    print(f"              - Không tìm thấy v1: '{hit.get('v1_text', '')}'")
                if not hit["v2_found"]:
                    print(f"              - Không tìm thấy v2: '{hit.get('v2_text', '')}'")

    print()
    print_separator()
    print("Evaluation complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    test_dirs = sorted([
        os.path.join(TEST_DATA_DIR, d)
        for d in os.listdir(TEST_DATA_DIR)
        if os.path.isdir(os.path.join(TEST_DATA_DIR, d))
    ])

    if not test_dirs:
        print(f"Không tìm thấy thư mục test nào trong {TEST_DATA_DIR}/")
        return

    print(f"Tìm thấy {len(test_dirs)} cặp test data.")
    print(f"LLM: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL}")
    print(f"Query: \"{BROAD_QUERY}\"")
    print()

    all_results = []
    for i, test_dir in enumerate(test_dirs, 1):
        name = os.path.basename(test_dir)
        gt_path = os.path.join(test_dir, "ground_truth.json")
        if not os.path.exists(gt_path):
            print(f"[{i}/{len(test_dirs)}] {name}: SKIP (không có ground_truth.json)")
            continue

        gt = load_ground_truth(test_dir)
        print(f"[{i}/{len(test_dirs)}] {name}: Đang chạy Diff Evaluation...")

        try:
            diff_result = evaluate_diff(test_dir, gt)
        except Exception as e:
            print(f"  ❌ Diff failed: {e}")
            diff_result = {"precision": 0, "recall": 0, "f1": 0, "found_dieu": [], "expected_dieu": [], "true_positives": [], "num_diffs": 0}

        print(f"  Diff: P={diff_result['precision']:.0f}% R={diff_result['recall']:.0f}% F1={diff_result['f1']:.0f}%")
        print(f"[{i}/{len(test_dirs)}] {name}: Đang chạy RAG Evaluation (cần LLM)...")

        try:
            rag_result = evaluate_rag(test_dir, gt)
        except Exception as e:
            print(f"  ❌ RAG failed: {e}")
            rag_result = {"change_recall": 0, "changes_detected": 0, "total_changes": len(gt.get("changes", [])), "grounding_score": 0, "latency": 0, "details": [], "raw_answer": ""}

        print(f"  RAG: Recall={rag_result['change_recall']:.0f}% Ground={rag_result['grounding_score']:.0f}% Time={rag_result['latency']:.1f}s")
        print()

        all_results.append({
            "name": name,
            "diff": diff_result,
            "rag": rag_result,
        })

    # In báo cáo tổng hợp
    print_report(all_results)

    # Lưu kết quả JSON
    output_path = "evaluation_results.json"
    serializable = []
    for r in all_results:
        entry = {
            "name": r["name"],
            "diff": {k: v for k, v in r["diff"].items()},
            "rag": {k: v for k, v in r["rag"].items() if k != "raw_answer"},
        }
        serializable.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả đã lưu vào: {output_path}")


if __name__ == "__main__":
    main()
