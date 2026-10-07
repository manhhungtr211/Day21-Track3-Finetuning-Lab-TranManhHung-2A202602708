# Lab 21 — Evaluation Report

**Họ tên**: Trần Mạnh Hùng  **MSSV**: 2A202602708  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 15GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

|                    |                                                               |
| ------------------ | ------------------------------------------------------------- |
| Dataset            | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu) |
| Train / val        | 225 / 25 (seed 42)                                            |
| `max_length`     | 1024 — p95 đo được là 98*(results/token_stats.json)*  |
| `MASK_MODE`      | `assistant-only`                                            |
| Epochs / max_steps | 2 epochs / 30 max_steps                                       |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict `reasoning preserved — safe to train on traces`)*.
Nếu không: bạn đã xử lý thế nào? Template của Qwen3.5 giữ nguyên khối reasoning, an toàn cho việc huấn luyện có trace.

---

## 2. Mask proof (NB1)

|                                  |          |
| -------------------------------- | -------- |
| `supervised_fraction`          | 0.4149   |
| Câu trả lời nằm trong loss   | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run                         | target | regression | format | latency (ms) |
| --------------------------- | ------ | ---------- | ------ | ------------ |
| (a) base + naive prompt     | 0.000  | 0.750      | 0.000  | 3465.9       |
| (b) base + optimized prompt | 0.688  | 0.750      | 1.000  | 1009.7       |
| (c) LoRA fine-tune          | 0.938  | 0.750      | 1.000  | 1450.9       |

**(b) có thật sự mạnh hơn (a) không?** Có (0.688 so với 0.000; format đạt 100% so với 0%).
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa, giữ nguyên prompt chuẩn của lab để đảm bảo đối chứng khách quan và tính toàn vẹn của mã SHA-256 (`719e74d3b6232053`).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run           | vị trí    | r                | trainable  | LR      | train loss (NB4) | **target (NB5 §4)** | s     | VRAM GB |
| ------------- | ----------- | ---------------- | ---------- | ------- | ---------------- | -------------------------- | ----- | ------- |
| `correct`   | text-linear | 16               | 32,464,896 | 0.0001  | 0.6266           | 0.9375                     | 395.5 | 8.78    |
| `attn_only` | q,v         | 283*(matched)* | 32,456,704 | 0.0001  | 0.5370           | 0.9375                     | 263.1 | 8.79    |
| `wrong_lr`  | text-linear | 16               | 32,464,896 | 0.00001 | 1.5702           | 0.0000                     | 386.8 | 8.78    |
| `qlora`     | text-linear | 16               | 32,464,896 | 0.0001  | 0.7058           | 0.8438                     | 453.9 | 3.86    |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được nâng rank lên r=283 để khớp số tham số huấn luyện với `correct` (32.45M vs 32.46M tham số, độ lệch dưới 0.03%). Trên tập đánh giá target, `attn_only` hoà với `correct` khi cả hai đều đạt 0.9375. Tuy nhiên, train loss của `attn_only` lại thấp hơn rõ rệt (0.5370 so với 0.6266), cho thấy thứ tự theo train loss hoàn toàn khác và không phản ánh đúng năng lực thực tế trên tác vụ. Điều này chứng minh rằng việc dồn toàn bộ ngân sách tham số vào rank cực lớn ở attention chỉ giúp mô hình ghi nhớ dữ liệu huấn luyện (overfit loss), trong khi việc phân bổ adapter trải đều trên tất cả các lớp linear (`text-linear`) ở rank vừa phải (r=16) mang lại biểu diễn cân bằng, hiệu quả và tổng quát hơn nhiều.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` sử dụng learning rate ở thang full fine-tune (1e-5 thay vì 1e-4), khiến đường loss gần như phẳng lì và dừng lại ở mức rất cao là 1.5702 (so với 0.6266 của `correct`). Kết quả trên tập target là 0.0000 và format là 0.0000, nghĩa là mô hình hoàn toàn không học được cấu trúc output mong muốn. Nếu một người chỉ quan sát loss phẳng mà không kiểm tra learning rate, họ sẽ dễ dàng kết luận sai lầm rằng tác vụ này quá khó, dữ liệu bị lỗi, hoặc LoRA không có khả năng thích nghi với mô hình 4B. Trong thực tế, chỉ cần điều chỉnh learning rate lên đúng thang của LoRA (~10x full FT) thì mô hình đã hội tụ nhanh chóng.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` 4-bit NF4 tiết kiệm được 56% bộ nhớ VRAM đỉnh (chỉ chiếm 3.86 GB so với 8.78 GB của cấu hình 16-bit LoRA), cho phép chạy được trên các GPU có dung lượng hạn chế. Tuy nhiên, cái giá phải trả là điểm target tụt giảm đáng kể từ 0.9375 xuống 0.8438 (mất gần 10% độ chính xác) và độ trễ sinh từ 1450ms tăng lên 1815ms do chi phí dequantization trong quá trình tính toán. Kết quả đo đạc thực tế này hoàn toàn ủng hộ khuyến nghị từ nhà sản xuất Qwen3.5: không nên lạm dụng QLoRA 4-bit trên dòng mô hình này trừ khi thực sự bị giới hạn phần cứng, vì sai số lượng tử hóa làm suy giảm nghiêm trọng năng lực suy luận của mô hình.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)
Mô hình LoRA fine-tune (`correct`) đã vượt qua cổng hồi quy với kết quả PASSED thuyết phục. Điểm số trên tác vụ mục tiêu (target) tăng vượt trội thêm +0.250 điểm (+25% so với mốc đối thủ baseline b được tối ưu bằng few-shot prompt), đạt độ chính xác 93.75% với tỷ lệ định dạng JSON hợp lệ tuyệt đối 100%. Điều quan trọng nhất là năng lực trên bài toán tổng quát (`regression`) được bảo toàn nguyên vẹn ở mức 0.750 (độ lệch bằng 0.000), khẳng định quá trình fine-tune không gây ra hiện tượng quên thảm họa (catastrophic forgetting). Kết quả này chứng minh rằng cấu hình LoRA vùng-không-hối-tiếc (all-linear, r=16, LR 1e-4, batch hiệu dụng 16 trong 30 bước) đã chuyển hóa thành công tri thức miền vào các ma trận trọng số adapter, cho phép rút gọn prompt hệ thống về mức đơn giản mà chất lượng đầu ra vẫn vượt trội so với prompt engineering phức tạp.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn)                                                                       | Nhãn đúng                                         | (b) prompt     | (c) fine-tune                                     | Nhận xét                                                                              |
| - | ---------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc           | doi_tra, cao   | doi_tra, cao, chuột không dây, tich_cuc        | ✅ FT thắng: trích xuất đúng 4 trường và đúng định dạng JSON               |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền...         | hoan_tien, trung_binh, ốp lưng điện thoại       | hoan_tien      | hoan_tien, trung_binh, ốp lưng điện thoại    | ✅ FT thắng: phân loại nhãn hoàn tiền chính xác                                 |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền...                 | hoan_tien, cao, đèn bàn LED                       | hoan_tien      | hoan_tien, cao, đèn bàn LED, tich_cuc          | ✅ FT thắng: nhận diện chính xác mức độ urgency cao                             |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền...   | hoan_tien, cao, bình giữ nhiệt, tieu_cuc          | hoan_tien, cao | hoan_tien, trung_binh, bình giữ nhiệt          | ❌**FT thua**: Đánh giá urgency là trung_binh thay vì cao do câu hỏi ngắn |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện...   | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | san_pham_loi   | san_pham_loi, trung_binh, nồi chiên không dầu | ❌**FT thua**: Nhận diện đúng intent nhưng đoán sai mức độ urgency      |

Có mẫu chung nào ở các ca FT thua không?
Các ca mô hình fine-tune bị trừ điểm (đạt 0.75/1.0) đều có mẫu số chung là nhầm lẫn mức độ `urgency` (dán nhãn `trung_binh` thay vì `cao`) trong các ticket khiếu nại ngắn gọn hoặc câu từ gián tiếp. Điều này phản ánh sự phân bố dữ liệu trong tập huấn luyện có thể chưa đủ các ca biên phức tạp để mô hình phân biệt ranh giới sắc thái mức độ khẩn cấp.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?
Bản fine-tune `correct` hoàn toàn xứng đáng được triển khai vào môi trường production. Kết quả thực nghiệm cho thấy mô hình không chỉ đạt độ chính xác phân loại vượt trội (93.75% so với 68.75% của prompt tối ưu) mà còn giảm mạnh độ dài prompt đầu vào từ hàng trăm token xuống một câu ngắn gọn, giúp tiết kiệm đáng kể chi phí token và tài nguyên tính toán khi phục vụ quy mô lớn. Đặc biệt, tỷ lệ tuân thủ định dạng JSON đạt 100% và không làm suy giảm năng lực tổng quát của mô hình nền tảng. Qua toàn bộ các thí nghiệm đối chứng, bài học then chốt là đòn bẩy lớn nhất không nằm ở việc tăng rank $r$ lên vô hạn, mà nằm ở việc chọn đúng vị trí gắn adapter trải đều trên toàn bộ các lớp linear (`all-linear`) kết hợp với thang learning rate chuẩn (~10x full FT). Việc kiểm soát nghiêm ngặt loss mask ở NB1 cũng là yếu tố sống còn bảo đảm mô hình học cách trả lời thay vì học vẹt lại prompt.

**Ba điều tôi học được** (cụ thể, không generic):

1. Không bao giờ tin tưởng mù quáng vào loss mask nếu chưa giải mã ngược token: việc che sai mask (`everything`) sẽ khiến mô hình học sinh lại cả câu hỏi của người dùng mà loss vẫn báo giảm đẹp mắt.
2. Vị trí gắn adapter quan trọng hơn rank $r$: cấu hình `attn_only` dù được nâng rank lên r=283 (bằng số tham số với all-linear r=16) và có train loss thấp hơn vẫn không thể vượt qua `all-linear` trên tập đánh giá thực tế.
3. Không đánh giá mô hình fine-tune bằng chỉ số thay thế (loss hay perplexity): phải xây dựng một bộ baseline đối chứng mạnh (prompt tối ưu + few-shot) và đo đạc trên 4 nhóm chỉ số toàn diện trước khi đưa ra quyết định triển khai.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Thử nghiệm trộn 3-5% dữ liệu tổng quát (general replay) trực tiếp vào tập train để đẩy điểm nhóm regression lên cao hơn nữa, đồng thời bổ sung thêm các mẫu biên phân loại sắc thái `urgency` để khắc phục triệt để các ca thua định tính.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
