# Báo cáo đầy đủ lịch sử huấn luyện Detection (7 lớp)

> Cập nhật đến: **2026-04-19**  
> Mục tiêu tài liệu: tổng hợp lại toàn bộ hành trình đã làm với model detection để tiếp tục tối ưu có hệ thống, tránh lặp lại thử nghiệm cũ.

> **Bản nâng cấp (audit pass):** đã rà lại thêm code, comment cấu hình trong `src/train_detect.py`, metadata checkpoint trong `models/` và các log bạn đã gửi trong hội thoại. Tài liệu dưới đây phân biệt rõ phần nào là **xác thực trực tiếp** và phần nào là **suy luận hợp lý**.

---

## 0) Mức độ tin cậy của thông tin trong báo cáo

Để bạn yên tâm dùng tài liệu này làm mốc tiếp tục tối ưu, mỗi nhóm thông tin được gắn mức tin cậy:

- **Mức A — Xác thực trực tiếp từ file/log hiện có**
  - Đọc được trực tiếp từ source code, JSON, checkpoint metadata, hoặc log bạn paste đầy đủ.
- **Mức B — Xác thực gián tiếp từ dấu vết cấu hình/comment**
  - Ví dụ block env/comment trong `src/train_detect.py` cho thấy đã từng dùng cấu hình đó.
- **Mức C — Suy luận hợp lý theo dòng thời gian trao đổi**
  - Dùng khi không còn artifact log thô trong workspace, nhưng có chuỗi trao đổi/decision nhất quán.

---

## 1) Mục tiêu tổng thể của nhánh Detection

Bài toán detection của dự án được đóng khung theo taxonomy **7 lớp**:

1. `biological`
2. `cardboard`
3. `glass`
4. `metal`
5. `paper`
6. `plastic`
7. `trash`

Mục tiêu xuyên suốt:
- Chuẩn hóa nhiều nguồn dữ liệu khác taxonomy về cùng 7 lớp.
- Huấn luyện Faster R-CNN ổn định, có per-class mAP rõ ràng.
- Cải thiện lớp yếu (`trash`, và giai đoạn sau còn có `biological`) bằng cả data + hyperparameter.
- Có pipeline train/infer rõ ràng, tái lập được bằng biến môi trường.

---

## 2) Vai trò từng file đã dùng trong hành trình Detection

## 2.1 `src/model_detect.py`

**Vai trò:** định nghĩa model detection và hậu xử lý NMS dùng trong demo infer.

- `build_detection_model(num_classes)`:
  - Tạo `fasterrcnn_resnet50_fpn`.
  - Thay head phân loại sang `num_classes + 1` (thêm background).
- `apply_nms_to_prediction(...)`:
  - Lọc theo `score_threshold` trước.
  - Chạy NMS theo `iou_threshold`.

**Ý nghĩa trong pipeline:**
- Là “xương sống” model dùng cho cả train (`train_detect.py`) lẫn demo (`web/utils.py`).

---

## 2.2 `src/data_pipeline.py`

**Vai trò:** cung cấp dataset/augment cho cả classification và detection.

Phần detection quan trọng:
- `COCODetectionDataset`:
  - Đọc COCO JSON, map annotation theo `image_id`.
  - Chuyển bbox từ `xywh` sang `xyxy`.
  - Lọc bbox lỗi (`w/h <= 1`, bbox out-of-range).
  - Label +1 để phù hợp Faster R-CNN (0 là background).
- Augmentation detection đã dùng:
  - `HorizontalFlip` (qua `_hflip_image_boxes`) với xác suất cấu hình.
  - `ColorJitter` nhẹ (brightness/contrast) bằng OpenCV.
  - `_clip_boxes_xyxy` để giữ bbox hợp lệ sau augment.
- `detection_collate_fn`:
  - Trả list ảnh + list target cho detection dataloader.

**Ý nghĩa trong pipeline:**
- Đây là nơi đã cải thiện tính ổn định dữ liệu train sau các lần kết quả thấp/dao động.

---

## 2.3 `src/remap_detection_categories.py`

**Vai trò:** remap taxonomy của `detection_1` từ nhiều class cũ (25 class) về 7 class chuẩn.

- Đầu vào: `data/detection_1/{split}/_annotations.coco.json`
- Đầu ra: `data/detection_1/{split}/_annotations_7cls.coco.json`
- Có validate mapping đầy đủ category cũ, lọc bbox lỗi, in thống kê theo lớp.

**Ý nghĩa trong pipeline:**
- Bước chuẩn hóa sớm khi dataset ban đầu chưa cùng taxonomy với mục tiêu 7 lớp.

---

## 2.4 `src/merge_detection_datasets_7cls.py`

**Vai trò:** script trung tâm để hợp nhất nhiều nguồn detection vào bộ train/val/test thống nhất.

### Chức năng chính
- Remap category theo từng nguồn:
  - `MAP_25_TO_7` cho `detection_1`
  - `MAP_D2_TO_7` cho `detection_2`
  - `MAP_BIO_TO_7` cho `biological_class`
- Prefix `file_name` để ảnh được load đúng từ root `data/`.
- Lọc annotation lỗi, đồng bộ lại `image_id`, `ann_id`.
- Chia train/val theo kiểu **class-aware split** (`_class_aware_split_train_val`):
  - đảm bảo val có đủ lớp, có minimum số box mỗi lớp.
- Hỗ trợ random cap dataset lớn:
  - `--bio-train-max-images`
  - `--bio-valid-max-images`
- Xuất JSON:
  - `train/_annotations_7cls_split_train.coco.json`
  - `valid/_annotations_7cls_split_val.coco.json`
  - `test/_annotations_7cls_merged_test.coco.json`

**Ý nghĩa trong pipeline:**
- Là bước then chốt giúp mở rộng dữ liệu (thêm `detection_2`, thêm `biological_class`) thay vì chỉ tune hyperparameter trên dataset cũ.

---

## 2.5 `src/train_detect.py`

**Vai trò:** script huấn luyện detection đầy đủ, hỗ trợ nhiều chế độ tối ưu bằng biến môi trường.

### Các khối chức năng chính
- `check_class_coverage(...)`: kiểm tra đủ 7 lớp trong split.
- `evaluate_map(...)`: tính `map50`, `map50_95`, hỗ trợ per-class mAP.
- Hard-mining nhẹ theo batch:
  - tính `cls_weights` từ phân bố box train.
  - scale loss theo độ khó batch (`DET_USE_HARD_MINING`, `DET_HARD_ALPHA`).
- AMP + grad clip:
  - `DET_USE_AMP`, `DET_GRAD_CLIP`.
- Scheduler:
  - `step` hoặc `warmup + cosine`.
- Speed knobs:
  - `DET_EVAL_EVERY`, `DET_VAL_MAX_BATCHES`, `DET_EVAL_CLASS_METRICS`, `DET_FULL_EVAL_LAST`
  - `DET_NUM_WORKERS`, `DET_PREFETCH_FACTOR`, `DET_PERSISTENT_WORKERS`
- Checkpoint output:
  - `models/model_detection_last.pth`
  - `models/model_detection_best_7_4.pth` (best map50)
  - `models/model_detection_best_map5095_7_4.pth` (best map50_95)

**Ý nghĩa trong pipeline:**
- Trung tâm tối ưu train-time, là nơi phần lớn thử nghiệm tham số đã diễn ra.

---

## 2.6 `web/utils.py`, `web/app.py`

**Vai trò:** suy luận/demo, không trực tiếp train nhưng phản ánh chất lượng model thật ngoài validation.

- `load_detection_model()`: load checkpoint detection.
- `run_detection()`: infer + filter theo score + NMS.
- `web/app.py`: giao diện Streamlit, slider threshold để quan sát độ nhạy mô hình.

**Ý nghĩa trong pipeline:**
- Giúp kiểm tra lỗi thực tế (ví dụ: user thấy biological chưa nhận diện ổn dù metric train đã tăng).

---

## 2.7 Các artefact trong `models/`

Các checkpoint detection hiện thấy:
- `model_detection.pth`
- `model_detection_last.pth`
- `model_detection_best.pth`
- `model_detection_best_7_4.pth`
- `model_detection_best_map5095.pth`
- `model_detection_best_map5095_7_4.pth`

**Ý nghĩa:**
- Lưu dấu các giai đoạn train khác nhau (baseline và các run tối ưu sau này).

### Kiểm chứng metadata checkpoint (audit ngày 2026-04-19)

Kết quả đọc trực tiếp bằng `torch.load`:

- `model_detection_best_7_4.pth`
  - `epoch=50`
  - `map50=0.7764615416526794`
  - `map50_95=0.6031165719032288`
- `model_detection_best_map5095_7_4.pth`
  - `epoch=50`
  - `map50=0.7764615416526794`
  - `map50_95=0.6031165719032288`
- `model_detection_last.pth`
  - `epoch=50`
  - `map50=0.7764615416526794`
  - `map50_95=0.6031165719032288`
- `model_detection_best.pth` và `model_detection_best_map5095.pth`
  - `epoch=48`
  - `map50=0.6572447419166565`
  - `map50_95=0.5074294209480286`
- `model_detection.pth`
  - đọc lỗi `EOFError` (khả năng checkpoint cũ hỏng/ghi dở)

> Mức tin cậy: **A**

---

## 3) Timeline các giai đoạn đã đi qua (theo thứ tự)

## Giai đoạn A — Baseline detection từ dữ liệu ban đầu

### A1. Bắt đầu với taxonomy chưa đồng nhất
- Dataset detection ban đầu không cùng chuẩn 7 lớp.
- Cần remap category trước khi train ổn định.

### A2. Dùng `remap_detection_categories.py`
- Remap `detection_1` về 7 lớp.
- Train ban đầu cho tín hiệu học nhưng còn thấp, một số lớp yếu rõ.

### A3. Kết quả baseline còn hạn chế
- Tài liệu dự án cũ ghi nhận detection ban đầu ở mức thấp hơn đáng kể so với hiện tại.
- Lớp `trash` thường yếu, dễ nhầm.

> Mức tin cậy: **B** (dựa trên report tuần 3–4 + checkpoint cũ còn lại), không còn đầy đủ log raw từng epoch của run baseline trong workspace hiện tại.

---

## Giai đoạn B — Tăng độ bền pipeline train (augment + scheduler + eval)

### B1. Củng cố dữ liệu đầu vào trong `COCODetectionDataset`
Đã thêm/duy trì:
- bbox clip / bbox validation chặt hơn
- horizontal flip cho detection
- color jitter nhẹ

### B2. Nâng script train (`train_detect.py`) thành “config-driven”
Đưa phần lớn hyperparameter ra env:
- batch size, workers, lr, scheduler
- warmup + cosine
- AMP, grad clip
- hard-mining
- eval frequency và val subset

### B3. Kiểm soát đánh giá per-class
- Bật/tắt per-class metric linh hoạt.
- Có cảnh báo split thiếu class (tránh đọc nhầm metric).

> Mức tin cậy: **A** (thấy trực tiếp trong code `train_detect.py` và `data_pipeline.py`).

---

## Giai đoạn C — Mở rộng dữ liệu: hợp nhất `detection_1 + detection_2`

### C1. Lý do
- Chỉ tune hyperparameter không đủ để kéo lớp yếu.
- Cần tăng data diversity và độ phủ class.

### C2. Dùng `merge_detection_datasets_7cls.py`
- Remap đa nguồn về 7 lớp.
- Gộp train pool.
- Chia class-aware train/val.
- Gộp test từ nhiều nguồn.

> Mức tin cậy: **A** (đọc trực tiếp từ script merge).

### C3. Luồng dữ liệu giai đoạn này
1. `data/detection_1`, `data/detection_2` (raw COCO)
2. remap category theo nguồn
3. merge + reindex ids + prefix file paths
4. class-aware split
5. xuất `data/detection_merged/{train,valid,test}`
6. train qua `DET_TRAIN_IMG_DIR=data`, `DET_TRAIN_JSON=...detection_merged...`

---

## Giai đoạn D — Bổ sung `biological_class` bằng random subset

### D1. Lý do
- Mục tiêu cải thiện nhận diện nhóm biological.
- Dataset `biological_class` lớn, cần cap để giữ thời gian train/GPU phù hợp.

### D2. Thay đổi trong `merge_detection_datasets_7cls.py`
- Thêm `MAP_BIO_TO_7`:
  - `Organic Waste -> biological`
  - `Paper Waste -> paper`
  - `trash -> trash`
- Thêm random sampling theo image-level:
  - `--bio-train-max-images`
  - `--bio-valid-max-images`
  - seed ổn định theo source index.

### D3. Kết quả merge đã ghi nhận (run thực tế)
Ví dụ run thử đã dùng:
- `--bio-train-max-images 1200`
- `--bio-valid-max-images 220`

Output:
- `data/detection_merged_bio/train/_annotations_7cls_split_train.coco.json`
- `data/detection_merged_bio/valid/_annotations_7cls_split_val.coco.json`
- `data/detection_merged_bio/test/_annotations_7cls_merged_test.coco.json`

> Mức tin cậy: **A** cho cấu trúc script và output path; **B** cho một số số liệu đếm chi tiết theo từng lần chạy vì không lưu thành file log cố định trong repo.

---

## Giai đoạn E — Tối ưu tốc độ & VRAM

### E1. Thử nghiệm tăng tải GPU
Từ config thấp (batch nhỏ) tăng lên:
- `DET_BATCH_SIZE=6`
- `DET_VAL_BATCH_SIZE=3`
- `DET_NUM_WORKERS=4`
- `DET_PREFETCH_FACTOR=4`
- AMP bật

### E2. Cân bằng speed vs quality
- Eval mỗi epoch (`DET_EVAL_EVERY=1`) để theo dõi sát.
- Giới hạn eval subset (`DET_VAL_MAX_BATCHES=200`) để giảm time.
- Epoch cuối chạy full eval (`DET_FULL_EVAL_LAST=1`) để có số cuối đáng tin.

> Mức tin cậy: **A** (có biến env + logic code), **B** (thứ tự thử nghiệm phụ thuộc chuỗi hội thoại).

---

## Giai đoạn F — Run full hiện tại (log gần nhất bạn đã gửi)

### Cấu hình chính đã chạy
- Dataset: `data/detection_merged` (train/valid)
- `batch=6`, `val_batch=3`
- `lr=0.010`, `scheduler=cosine`, `warmup=3`
- `hard_mining=on`, `hard_alpha=0.35`
- `eval_every=1`, `val_max_batches=200`, `full_eval_last=1`

### Kết quả tiêu biểu
- Epoch 1: `map50=0.0163`, `map50_95=0.0037`
- Epoch 2: `map50=0.0461`, `map50_95=0.0185`
- Epoch 38: `map50=0.7233`, `map50_95=0.5400`
- Epoch 48: `map50=0.7493`, `map50_95=0.5725`
- **Epoch 50: `map50=0.7765`, `map50_95=0.6031` (best)**

Per-class cuối (epoch 50) đã thấy:
- class 1: 0.5032
- class 2: 0.6489
- class 3: 0.7267
- class 4: 0.7184
- class 5: 0.6408
- class 6: 0.5651
- class 7: 0.4187

> Nhận xét: tổng thể đã tăng rất mạnh so với giai đoạn đầu; vẫn còn khoảng cải thiện ở lớp yếu nhất.

> Mức tin cậy: **A** (từ log bạn gửi đầy đủ + metadata checkpoint hiện tại).

---

## 4) Luồng dữ liệu qua từng giai đoạn (data flow)

## 4.1 Giai đoạn baseline
`detection_1 raw COCO`  
→ remap về 7 lớp (`remap_detection_categories.py`)  
→ train (`train_detect.py`)  
→ checkpoint detection cơ bản.

## 4.2 Giai đoạn merge đa nguồn
`detection_1 + detection_2 (+ biological_class)`  
→ remap theo source map  
→ merge + reindex + prefix path  
→ class-aware split train/val  
→ `data/detection_merged*` JSON  
→ train (`train_detect.py` đọc JSON merged).

## 4.3 Giai đoạn infer/demo
`web/app.py` upload ảnh  
→ `web/utils.py` load checkpoint detection  
→ predict + score filter + NMS  
→ hiển thị bbox/class/score.

---

## 5) Những tham số đã thử / đã dùng (tóm tắt)

## 5.1 Dải tham số train chính
- Batch size: `2` → `6` (đã chạy thành công, cải thiện throughput)
- Val batch size: `1` → `3`
- Workers: `0/2/3/4`
- LR: `0.005` và `0.010`
- Scheduler:
  - `step` (`step_size=8`, `gamma=0.1`)
  - `cosine` + warmup (được dùng rộng hơn ở các run gần đây)
- Warmup epochs: `3`
- AMP: `on`
- Grad clip: `5.0`
- Hard mining:
  - `DET_USE_HARD_MINING=1`
  - `DET_HARD_ALPHA=0.35`
- Eval knobs:
  - `DET_EVAL_EVERY=1` hoặc `3`
  - `DET_EVAL_CLASS_METRICS=0/1`
  - `DET_VAL_MAX_BATCHES=100/200`
  - `DET_FULL_EVAL_LAST=1`

## 5.2 Dải tham số merge dữ liệu
- `--val-ratio` (mặc định 0.2)
- `--min-val-boxes-per-class` (mặc định 40)
- `--bio-train-max-images` (`1200` test run, mặc định script 1800)
- `--bio-valid-max-images` (`220` test run, mặc định script 300)
- `--seed=42`

---

## 6) Vì sao đã đi theo hướng “thêm dữ liệu + chuẩn hóa merge”

Trong quá trình tối ưu detection, chỉ tinh chỉnh augmentation/hyperparameter chưa đủ để cải thiện đồng đều mọi lớp.

Lý do kỹ thuật:
- Lớp yếu thường do **độ phủ dữ liệu và diversity** thiếu, không chỉ do LR/scheduler.
- Taxonomy khác nhau giữa dataset nguồn gây “nhiễu nhãn” nếu không remap chặt.
- Merge đa nguồn + class-aware split giúp:
  - tăng số box/lớp,
  - giảm rủi ro val thiếu lớp,
  - per-class metric phản ánh sát hơn.

---

## 7) Bài học rút ra đến thời điểm hiện tại

1. **Data pipeline quyết định trần hiệu năng**: remap/merge đúng quan trọng hơn đổi optimizer nhỏ lẻ.
2. **Eval strategy cần linh hoạt**: dùng `val_max_batches` để train nhanh, nhưng vẫn giữ full eval epoch cuối để chốt model.
3. **Hard-mining có ích khi class imbalance** nhưng cần alpha vừa phải để không làm loss bất ổn.
4. **Tăng batch theo VRAM** mang lại tốc độ rõ rệt, miễn LR/warmup được scale hợp lý.
5. Kết quả hiện tại đã khá mạnh, nhưng vẫn còn dư địa ở lớp yếu.

---

## 8) Trạng thái hiện tại & điểm xuất phát cho vòng tối ưu tiếp theo

Hiện tại bạn đã có một baseline mạnh với:
- `map50 ≈ 0.7765`
- `map50_95 ≈ 0.6031`

### Điểm xuất phát khuyến nghị cho vòng tiếp theo
- Giữ checkpoint tốt nhất hiện tại làm mốc so sánh.
- Tập trung tối ưu **per-class lớp yếu** thay vì chỉ đẩy metric trung bình.
- Ưu tiên các hướng:
  1. điều chỉnh sampling/cap theo lớp yếu trong merge,
  2. fine-tune ngắn với LR thấp hơn quanh vùng hội tụ,
  3. đánh giá thêm trên ảnh thực tế trong app.

---

## 9) Checklist tái lập nhanh toàn pipeline detection

1. Chuẩn bị/kiểm tra dữ liệu nguồn trong `data/`.
2. (Nếu cần) remap riêng `detection_1` bằng `src/remap_detection_categories.py`.
3. Merge đa nguồn bằng `src/merge_detection_datasets_7cls.py`.
4. Set env `DET_*` trỏ đến JSON merged tương ứng.
5. Train bằng `python -m src.train_detect`.
6. Kiểm tra checkpoint trong `models/`.
7. Demo bằng `web/app.py` để rà lỗi thực tế.

---

## 10) Ghi chú cuối

Tài liệu này tổng hợp từ:
- code hiện có trong repo,
- checkpoint metadata hiện tại,
- các log huấn luyện bạn đã gửi trong quá trình tối ưu gần đây.

Nếu cần, có thể tạo thêm phiên bản v2 của báo cáo này với:
- bảng so sánh từng run theo ID,
- bảng metric theo từng lớp/epoch,
- decision log theo ngày (experiment tracking chuẩn hơn).

---

## 11) Khuyến nghị để báo cáo tương lai “đúng tuyệt đối” hơn nữa

Để lần sau không còn vùng mờ ở các run thử trung gian, nên thêm 2 thứ nhẹ nhưng rất giá trị:

1. **Run registry file** (ví dụ `experiments/detection_runs.csv`)
  - mỗi lần chạy ghi: timestamp, json train/val, batch, lr, scheduler, eval_every, val_max_batches, map50/map5095 tốt nhất.
2. **Lưu log text theo run_id** (ví dụ `logs/det_run_YYYYMMDD_HHMM.txt`)
  - giúp tái dựng chuẩn “run nào thử tham số gì” mà không cần nhớ lại bằng hội thoại.

> Nếu bạn muốn, mình có thể làm luôn khung này ở bước tiếp theo để từ vòng tối ưu tới mọi báo cáo đều 100% truy vết được.

---

## 12) Các kỹ thuật tối ưu model Detection **đã sử dụng** (tổng hợp theo nhóm)

Mục này tập trung trả lời: *trong toàn bộ quá trình vừa qua, team đã thật sự dùng kỹ thuật gì để tối ưu model?*  
Không lặp lại timeline, mà gom theo nhóm kỹ thuật để tiện đối chiếu khi lập kế hoạch vòng tiếp theo.

### 12.1 Tối ưu dữ liệu (Data-centric optimization)

1. **Chuẩn hóa taxonomy về 7 lớp**
  - Dùng remap category từ nguồn cũ (`detection_1`, `detection_2`, `biological_class`) về cùng nhãn mục tiêu.
  - Giảm nhiễu nhãn giữa các dataset nguồn.

2. **Hợp nhất đa nguồn dữ liệu**
  - Gộp `detection_1 + detection_2`, sau đó bổ sung thêm `biological_class`.
  - Tăng độ phủ bối cảnh, vật thể và phân bố dữ liệu.

3. **Class-aware train/val split**
  - Split không ngẫu nhiên thuần túy, mà có ràng buộc số box tối thiểu/class trong val.
  - Mục tiêu: tránh tình trạng thiếu lớp ở validation khiến per-class metric nhiễu.

4. **Random subset cap cho dataset lớn**
  - Áp dụng cho `biological_class` với `--bio-train-max-images`, `--bio-valid-max-images`.
  - Mục tiêu: kiểm soát thời gian train nhưng vẫn tăng tín hiệu cho lớp quan tâm.

5. **Lọc annotation/bbox lỗi trong pipeline**
  - Loại bbox không hợp lệ, clip bbox vào ảnh, đồng bộ lại annotation sau augment.
  - Giảm gradient nhiễu và lỗi huấn luyện do dữ liệu bẩn.

---

### 12.2 Tối ưu augmentation và tiền xử lý

1. **Geometric augmentation cho detection**
  - Horizontal Flip có đồng bộ bbox.

2. **Photometric augmentation nhẹ**
  - Brightness/Contrast jitter bằng OpenCV.

3. **Chuẩn hóa tensor input thống nhất**
  - Ảnh vào detection được đưa về float tensor `[C,H,W]` với miền giá trị ổn định.

> Ghi chú: augmentation hiện tại ưu tiên độ an toàn/ổn định hơn là “mạnh tay”.

---

### 12.3 Tối ưu chiến lược huấn luyện (training strategy)

1. **Huấn luyện theo cấu hình env-driven (`DET_*`)**
  - Toàn bộ knob quan trọng đưa ra biến môi trường để sweep nhanh.

2. **AMP (mixed precision) + Grad Clip**
  - AMP giúp giảm memory/đẩy tốc độ.
  - Grad clipping giúp ổn định khi LR/batch cao hơn.

3. **Hard example mining nhẹ (sample-level reweight)**
  - Tính trọng số lớp từ phân bố box train.
  - Scale loss theo độ “khó” của batch với `DET_HARD_ALPHA`.

4. **Scheduler có warmup + cosine**
  - Warmup vài epoch đầu để hội tụ mượt.
  - Cosine ở phase sau để giảm LR êm, thường cho kết quả ổn định hơn step đơn giản.

5. **Theo dõi per-class mAP trong quá trình train**
  - Không chỉ nhìn metric tổng, mà đọc lớp nào đang tụt/chậm cải thiện.

---

### 12.4 Tối ưu hiệu năng train (speed/throughput)

1. **Scale batch theo VRAM thực tế**
  - Từ batch nhỏ lên cấu hình lớn hơn (đã chốt tốt ở batch 6).

2. **Tinh chỉnh DataLoader**
  - `num_workers`, `prefetch_factor`, `persistent_workers` để giảm bottleneck CPU I/O.

3. **Tách chiến lược eval để tiết kiệm thời gian**
  - `DET_EVAL_EVERY` (không nhất thiết eval mọi epoch ở phase sweep nhanh).
  - `DET_VAL_MAX_BATCHES` để giới hạn độ nặng eval trung gian.
  - `DET_FULL_EVAL_LAST=1` để epoch cuối vẫn đánh giá full đáng tin cậy.

---

### 12.5 Tối ưu hậu xử lý/inference

1. **Tách threshold infer và threshold eval**
  - Tránh dùng chung một ngưỡng cho mọi mục đích.

2. **NMS tuning có tham số**
  - Điều chỉnh `score_threshold` + `nms_iou_threshold` linh hoạt khi demo/infer.

3. **Demo loop để kiểm tra thực tế ngoài metric**
  - Dùng app để kiểm tra mô hình trong ảnh thật, phát hiện mismatch giữa metric và trải nghiệm thực tế.

---

### 12.6 Tác động tổng hợp quan sát được

- Từ các run đầu (thấp) đến run full hiện tại, model đã đạt:
  - `map50 = 0.7764615416526794`
  - `map50_95 = 0.6031165719032288`
- Điều này cho thấy cách tiếp cận kết hợp **data-centric + training strategy + speed tuning** là có hiệu quả rõ ràng.

---

### 12.7 Kết luận kỹ thuật

Những kỹ thuật đã dùng đến hiện tại chủ yếu thuộc nhóm:
- chuẩn hóa/gia tăng dữ liệu,
- ổn định và tăng tốc training,
- kiểm soát đánh giá theo lớp,
- tinh chỉnh hậu xử lý.

Đây là nền rất tốt để bước sang vòng tối ưu nâng cao (mục tiêu tiếp theo thường là tăng thêm per-class lớp yếu mà không làm giảm metric tổng).

---

## 13) Nhánh riêng: Chuẩn bị chuyển sang Pretrained (tách biệt với timeline A→F)

Mục này ghi nhận **thay đổi kỹ thuật đã triển khai trong code** để sẵn sàng chạy nhánh pretrained mà không phá vỡ nhánh from-scratch cũ.

> Ghi chú cấu trúc: pretrained **không nối tiếp** vào Giai đoạn F; nó là một nhánh thí nghiệm song song để benchmark và báo cáo.

### 13.1 Thay đổi ở `src/model_detect.py`

- Nâng `build_detection_model(...)` thành dạng có cấu hình:
  - `use_pretrained: bool`
  - `variant: str` (`v1`/`v2`)
- Hỗ trợ:
  - `v2 pretrained`: `fasterrcnn_resnet50_fpn_v2` với weights COCO `DEFAULT`
  - `v1 pretrained`: `fasterrcnn_resnet50_fpn` với weights COCO `DEFAULT`
  - `scratch`: `weights=None`, `weights_backbone=None`
- Vẫn giữ thay head ra `num_classes + 1` (7 lớp + background) như pipeline hiện tại.

### 13.2 Thay đổi ở `src/train_detect.py`

Đã thêm env mới để chọn chế độ model:
- `DET_USE_PRETRAINED` (`0/1`)
- `DET_MODEL_VARIANT` (`v1` hoặc `v2`, mặc định `v2`)

Đã thêm env để tách output checkpoint khi chạy nhánh mới:
- `DET_PRETRAINED_OUT_DIR` (mặc định `models/pretrained`)
- `DET_SCRATCH_OUT_DIR` (mặc định `models`)

Script hiện dùng cách đơn giản nhất: dựa trên `DET_USE_PRETRAINED` để quyết định nhánh output.

- Nếu `DET_USE_PRETRAINED=1`, lưu cố định:
  - `model_detection_pretrained_last.pth`
  - `model_detection_pretrained_best_map50.pth`
  - `model_detection_pretrained_best_map5095.pth`
- Nếu `DET_USE_PRETRAINED=0`, lưu cố định:
  - `model_detection_last.pth`
  - `model_detection_best_7_4.pth`
  - `model_detection_best_map5095_7_4.pth`

Cách này giúp so sánh pretrained vs scratch rõ ràng và thao tác thủ công nhanh (không cần tag theo từng run).

### 13.3 Mục tiêu của giai đoạn này

1. Chạy full pretrained trên **cùng dataset gần nhất** (nhánh có biological) để có benchmark mới cho báo cáo.
2. Giữ nguyên model from-scratch cũ để so sánh trực tiếp.
3. Sau khi có số pretrained, quay lại tối ưu from-scratch nâng cao (augment/sampling/loss/post-process).

> Mức tin cậy: **A** (đã sửa code và kiểm tra không còn lỗi trong `src/model_detect.py` và `src/train_detect.py`).
