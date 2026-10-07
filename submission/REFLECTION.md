# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là việc `attn_only` được nâng rank lên r=283 (cùng số tham số với all-linear r=16) có train loss thấp hơn hẳn `correct` (0.5370 vs 0.6266), nhưng khi đo trên tập target thì chỉ hoà (0.9375). Điều này chứng minh train loss thấp chỉ là do overfit và ghi nhớ, không đồng nghĩa với khả năng tổng quát hóa tốt hơn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở NB4 khi phải huấn luyện 3 mô hình đối chứng liên tiếp (~45-50 phút). Đúng như dự đoán từ tài liệu phần cứng, đây là phần tốn tài nguyên GPU nhất trong toàn bộ lab.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng "chỉ cần train loss giảm và perplexity giảm là mô hình đã học tốt". Sau khi trực tiếp đo 4 nhóm chỉ số và thấy `wrong_lr` hay `attn_only` có loss giảm nhưng vẫn thất bại trên tác vụ thực tế, tôi hiểu rằng chỉ số thay thế hoàn toàn không đáng tin cậy nếu không đo trực tiếp trên bài toán mục tiêu.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích các cơ chế loss mask, cấu hình tier, và phân tích các tham số trong `LoraSpec`. AI đôi khi đưa ra lời khuyên dùng QLoRA 4-bit theo thói quen cũ mà không nhận ra rằng trên dòng kiến trúc Qwen3.5, nhà sản xuất đã khuyến nghị dùng 16-bit LoRA để tránh lỗi lượng tử hóa.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm là xây dựng một Baseline (b) thật mạnh bằng Prompt Engineering và Few-shot, sau đó đóng băng tập đánh giá để làm mốc so sánh. Nếu prompt engineering chưa khai thác hết mà vội vàng fine-tune thì vừa tốn kém chi phí vừa không biết chắc mô hình có thật sự mang lại giá trị vượt trội hay không.
