import json
import re
from typing import List, Dict, Any
from src.generation.llm_client import LLMClient

class MetricsEvaluator:
    """
    Module đánh giá chất lượng RAG và Text Diff Engine dựa trên LLM-as-a-Judge.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = (
            "Bạn là một Chuyên gia Đánh giá (Evaluation Judge) hệ thống AI so sánh văn bản pháp lý.\n"
            "Nhiệm vụ của bạn là đối chiếu kết quả đầu ra của hệ thống (System Output) với Danh sách thay đổi thực tế (Ground Truth Changes) để xác định xem hệ thống đã nhận diện đúng, bỏ sót hay báo cáo sai các thay đổi.\n\n"
            "Hãy trả về kết quả dưới định dạng JSON duy nhất, tuyệt đối không kèm theo bất kỳ văn bản giải thích nào khác bên ngoài khối JSON. Định dạng JSON yêu cầu như sau:\n"
            "{\n"
            '  "matched_changes": [\n'
            "    {\n"
            '      "gt_id": "c1",\n'
            '      "is_captured": true,\n'
            '      "citation_correct": true,\n'
            '      "explanation": "Giải thích ngắn gọn tại sao khớp hoặc không khớp"\n'
            "    }\n"
            "  ],\n"
            '  "extra_changes": [\n'
            "    {\n"
            '      "description": "Mô tả thay đổi dư thừa/sai lệch mà hệ thống tự đưa ra nhưng không có trong Ground Truth",\n'
            '      "location": "Vị trí được báo cáo"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Trích xuất và parse JSON từ phản hồi của LLM."""
        try:
            # Tìm kiếm khối JSON trong ```json ... ``` hoặc tự do
            match = re.search(r"({.*})", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(response_text)
        except Exception as e:
            print(f"[Evaluator] Không thể parse JSON từ LLM response: {e}")
            print(f"[Evaluator] Phản hồi gốc: {response_text}")
            return {"matched_changes": [], "extra_changes": []}

    def _build_evaluation_prompt(self, system_output: str, ground_truth_changes: List[Dict[str, Any]], context_type: str) -> str:
        gt_str = json.dumps(ground_truth_changes, ensure_ascii=False, indent=2)
        return (
            f"=== LOẠI ĐÁNG GIÁ: {context_type} ===\n\n"
            f"=== SYSTEM OUTPUT (KẾT QUẢ CỦA HỆ THỐNG) ===\n"
            f"{system_output}\n\n"
            f"=== GROUND TRUTH CHANGES (THỰC TẾ CÓ SẴN) ===\n"
            f"{gt_str}\n\n"
            f"Hãy thực hiện đối chiếu tỉ mỉ. Phân tích xem các thay đổi trong Ground Truth đã xuất hiện trong System Output chưa.\n"
            f"Lưu ý: System Output là kết quả từ `{context_type}` nên câu chữ có thể khác biệt nhưng ý nghĩa so sánh/thay đổi phải khớp.\n"
            f"Nếu System Output báo cáo một thay đổi hoàn toàn không có hoặc không tương đồng với thay đổi nào trong Ground Truth, hãy đưa vào 'extra_changes'.\n"
            f"Chỉ trả ra duy nhất định dạng JSON."
        )

    def evaluate_diff(self, diff_text: str, ground_truth_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Đánh giá Text Diff Engine.
        """
        if not ground_truth_changes:
            return {
                "precision": 1.0 if not diff_text.strip() else 0.0,
                "recall": 1.0,
                "f1": 1.0 if not diff_text.strip() else 0.0,
                "tp": 0, "fp": 0, "fn": 0,
                "citation_accuracy": 1.0,
                "details": {}
            }

        prompt = self._build_evaluation_prompt(diff_text, ground_truth_changes, "Text Diff Engine (Output dạng raw DIFF)")
        response = self.llm.generate_response(prompt=prompt, system_prompt=self.system_prompt)
        judge_res = self._parse_json_response(response)

        return self._calculate_metrics(judge_res, ground_truth_changes)

    def evaluate_rag(self, rag_response: str, ground_truth_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Đánh giá RAG Pipeline (Retrieval & Generation).
        """
        if not ground_truth_changes:
            return {
                "precision": 1.0 if not rag_response.strip() else 0.0,
                "recall": 1.0,
                "f1": 1.0 if not rag_response.strip() else 0.0,
                "tp": 0, "fp": 0, "fn": 0,
                "citation_accuracy": 1.0,
                "details": {}
            }

        prompt = self._build_evaluation_prompt(rag_response, ground_truth_changes, "RAG Pipeline (Văn bản trả lời của LLM)")
        response = self.llm.generate_response(prompt=prompt, system_prompt=self.system_prompt)
        judge_res = self._parse_json_response(response)

        return self._calculate_metrics(judge_res, ground_truth_changes)

    def _calculate_metrics(self, judge_res: Dict[str, Any], ground_truth_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        matched = {item["gt_id"]: item for item in judge_res.get("matched_changes", [])}
        
        tp = 0
        fn = 0
        correct_citations = 0
        total_evaluable_citations = 0

        for gt in ground_truth_changes:
            gt_id = gt["id"]
            if gt_id in matched and matched[gt_id].get("is_captured", False):
                tp += 1
                total_evaluable_citations += 1
                if matched[gt_id].get("citation_correct", False):
                    correct_citations += 1
            else:
                fn += 1

        extra_changes = judge_res.get("extra_changes", [])
        fp = len(extra_changes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        citation_accuracy = correct_citations / total_evaluable_citations if total_evaluable_citations > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "citation_accuracy": citation_accuracy,
            "details": judge_res
        }
