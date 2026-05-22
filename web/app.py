import streamlit as st
from PIL import Image

from web.utils import (
	DEFAULT_CLASS_NAMES,
	PLOTLY_AVAILABLE,
	build_interactive_bbox_figure,
	classify_boxes,
	draw_predictions,
	fuse_detection_and_classification,
	load_classification_model,
	load_detection_model,
	prepare_image,
	run_detection,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str, num_classes: int):
	"""Load model 1 lần cho mỗi checkpoint để app rerun nhanh hơn."""
	return load_detection_model(
		model_path=model_path,
		num_classes=num_classes,
	)


@st.cache_resource(show_spinner=False)
def get_cached_classification_model(model_path: str):
	"""Load classification model 1 lần để vote lại class từ bbox crop."""
	return load_classification_model(model_path=model_path)


st.set_page_config(page_title="Trash Detection Demo", layout="wide")
st.title("🗑️ Trash Detection Demo")
st.caption("Upload 1 ảnh để kiểm tra bounding box và class dự đoán của model detection.")


with st.sidebar:
	st.header("Cấu hình suy luận")
	num_classes = 7  # cố định theo dataset/checkpoint hiện tại
	infer_mode = st.selectbox(
		"Chế độ",
		options=["Detection only", "Detection + Classification"],
		index=0,
	)
	det_ckpt_presets = {
		"7_4 best (scratch)": "models/model_detection_best_7_4.pth",
		"7_4 best map50_95": "models/model_detection_best_map5095_7_4.pth",
		"Pretrained best map50_95": "models/pretrained/model_detection_pretrained_best_map5095.pth",
		"Pretrained best map50": "models/pretrained/model_detection_pretrained_best_map50.pth",
		"Custom path": "__custom__",
	}
	selected_det_preset = st.selectbox(
		"Detection checkpoint preset",
		options=list(det_ckpt_presets.keys()),
		index=0,
	)

	if selected_det_preset == "Custom path":
		model_path = st.text_input("Checkpoint path", value="models/model_detection_best_7_4.pth")
	else:
		model_path = det_ckpt_presets[selected_det_preset]
		st.text_input("Checkpoint path", value=model_path, disabled=True)
	cls_model_path = st.text_input("Classification checkpoint", value="models/model_classification_best.pth")
	score_threshold = st.slider("Score threshold", min_value=0.01, max_value=0.95, value=0.10, step=0.01)
	nms_iou_threshold = st.slider("NMS IoU threshold", min_value=0.10, max_value=0.95, value=0.40, step=0.01)
	cls_override_threshold = st.slider(
		"Ngưỡng override của classification",
		min_value=0.30,
		max_value=0.95,
		value=0.65,
		step=0.01,
		disabled=(infer_mode != "Detection + Classification"),
	)
	st.markdown("**Số class (không tính background):** 7 *(cố định theo checkpoint)*")

uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is None:
	st.info("Hãy upload một ảnh để bắt đầu demo.")
	st.stop()

try:
	model, device, resolved_model_path = get_cached_model(
		model_path=model_path,
		num_classes=int(num_classes),
	)
except Exception as exc:
	st.error(f"Không load được model: {exc}")
	st.stop()

cls_model = None
cls_device = None
idx_to_class = None
resolved_cls_model_path = None

if infer_mode == "Detection + Classification":
	try:
		cls_model, cls_device, resolved_cls_model_path, idx_to_class = get_cached_classification_model(
			model_path=cls_model_path,
		)
	except Exception as exc:
		st.warning(f"Không load được classification model, fallback về Detection only: {exc}")
		infer_mode = "Detection only"

image = Image.open(uploaded_file)
image_np, image_tensor = prepare_image(image)

prediction = run_detection(
	model=model,
	image_tensor=image_tensor,
	device=device,
	score_threshold=float(score_threshold),
	nms_iou_threshold=float(nms_iou_threshold),
)

cls_results = []
if infer_mode == "Detection + Classification" and cls_model is not None and idx_to_class is not None:
	cls_results = classify_boxes(
		image_np=image_np,
		prediction=prediction,
		model=cls_model,
		device=cls_device,
		idx_to_class=idx_to_class,
	)

rendered = draw_predictions(image_np=image_np, prediction=prediction, class_names=DEFAULT_CLASS_NAMES)

# Layout cố định: trái hiển thị ảnh, phải hiển thị thông tin + kết luận.
col1, col2 = st.columns([1.45, 1.0], gap="large")
with col1:
	st.subheader("Ảnh")
	tab_original, tab_pred = st.tabs(["Ảnh gốc", "Ảnh dự đoán"])
	with tab_original:
		st.image(image_np, channels="RGB", use_container_width=True)
	with tab_pred:
		num_boxes = int(prediction["boxes"].shape[0]) if "boxes" in prediction else 0
		interactive_rows = []
		if num_boxes > 0:
			for idx, (box, label, score) in enumerate(zip(prediction["boxes"], prediction["labels"], prediction["scores"])):
				det_idx = int(label.item()) - 1
				det_class = DEFAULT_CLASS_NAMES[det_idx] if 0 <= det_idx < len(DEFAULT_CLASS_NAMES) else f"cls_{int(label.item())}"
				det_score = float(score.item())
				x1, y1, x2, y2 = [float(v) for v in box.tolist()]

				if infer_mode == "Detection + Classification" and idx < len(cls_results):
					cls_class = cls_results[idx]["class"]
					cls_score = float(cls_results[idx]["score"])
					final_class, final_score, fusion_source = fuse_detection_and_classification(
						det_class=det_class,
						det_score=det_score,
						cls_class=cls_class,
						cls_score=cls_score,
						cls_override_threshold=float(cls_override_threshold),
					)
					hover_text = (
						f"bbox #{idx + 1}<br>"
						f"det: {det_class} ({det_score:.3f})<br>"
						f"cls: {cls_class} ({cls_score:.3f})<br>"
						f"final: {final_class} ({final_score:.3f})"
					)
				else:
					final_class, final_score, fusion_source = det_class, det_score, "det_only"
					hover_text = f"bbox #{idx + 1}<br>{final_class}: {final_score:.3f}"

				interactive_rows.append(
					{
						"id": idx + 1,
						"x1": x1,
						"y1": y1,
						"x2": x2,
						"y2": y2,
						"det_class": det_class,
						"det_score": det_score,
						"final_class": final_class,
						"final_score": final_score,
						"fusion_source": fusion_source,
						"hover_text": hover_text,
					}
				)

		if PLOTLY_AVAILABLE and len(interactive_rows) > 0:
			fig = build_interactive_bbox_figure(image_np=image_np, rows=interactive_rows)
			st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
			st.caption("Di chuột lên viền bbox để xem class + score.")
		else:
			st.image(rendered, channels="RGB", use_container_width=True)
			if not PLOTLY_AVAILABLE:
				st.caption("Chưa có plotly nên chưa bật được hover trực tiếp trên bbox.")

with col2:
	st.subheader("Thông tin dự đoán")
	num_boxes = int(prediction["boxes"].shape[0]) if "boxes" in prediction else 0
	st.write(f"- Device: `{device}`")
	st.write(f"- Checkpoint: `{resolved_model_path}`")
	st.write(f"- Chế độ: `{infer_mode}`")
	if infer_mode == "Detection + Classification" and resolved_cls_model_path is not None:
		st.write(f"- Classification checkpoint: `{resolved_cls_model_path}`")
	st.write(f"- Số bbox giữ lại sau threshold + NMS: **{num_boxes}**")

	st.subheader("Kết luận cuối cùng")

	if num_boxes > 0:
		rows = []
		top_class = None
		top_score = -1.0
		for idx, (box, label, score) in enumerate(zip(prediction["boxes"], prediction["labels"], prediction["scores"])):
			det_idx = int(label.item()) - 1
			det_class = DEFAULT_CLASS_NAMES[det_idx] if 0 <= det_idx < len(DEFAULT_CLASS_NAMES) else f"cls_{int(label.item())}"
			det_score = float(score.item())

			if infer_mode == "Detection + Classification" and idx < len(cls_results):
				cls_class = cls_results[idx]["class"]
				cls_score = float(cls_results[idx]["score"])
				final_class, final_score, fusion_source = fuse_detection_and_classification(
					det_class=det_class,
					det_score=det_score,
					cls_class=cls_class,
					cls_score=cls_score,
					cls_override_threshold=float(cls_override_threshold),
				)
			else:
				cls_class, cls_score = "-", 0.0
				final_class, final_score, fusion_source = det_class, det_score, "det_only"

			if final_score > top_score:
				top_score = final_score
				top_class = final_class

			x1, y1, x2, y2 = [round(float(v), 1) for v in box.tolist()]
			row = {
				"bbox_id": idx + 1,
				"det_class": det_class,
				"det_score": round(det_score, 4),
				"final_class": final_class,
				"final_score": round(final_score, 4),
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
			}
			if infer_mode == "Detection + Classification":
				row.update(
					{
						"cls_class": cls_class,
						"cls_score": round(cls_score, 4),
						"fusion": fusion_source,
					}
				)
			rows.append(row)
		st.dataframe(rows, use_container_width=True)

		if top_score >= 0.75:
			confidence_note = "độ tin cậy cao"
		elif top_score >= 0.5:
			confidence_note = "độ tin cậy trung bình"
		else:
			confidence_note = "độ tin cậy thấp (nên kiểm tra thêm)"

		st.success(
			f"Kết luận: ảnh này model dự đoán nổi bật nhất là **{top_class}** "
			f"(score={top_score:.3f}, {confidence_note})."
		)
	else:
		st.warning("Không có bbox nào vượt qua ngưỡng hiện tại. Hãy giảm score threshold.")
		st.info("Kết luận: chưa đủ bằng chứng để kết luận class từ ảnh hiện tại với ngưỡng đang chọn.")
