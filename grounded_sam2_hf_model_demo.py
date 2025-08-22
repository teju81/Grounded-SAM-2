import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def single_mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_grounding(processor, model, device, image, text, box_thr, text_thr, size=None):
    """
    size can be:
      - None (model default)
      - dict with BOTH keys: {"shortest_edge": 800, "longest_edge": 1536}
      - dict with BOTH keys: {"height": H, "width": W}
    """
    kwargs = dict(images=image, text=text, return_tensors="pt")
    if size is not None:
        kwargs["size"] = size

    inputs = processor(**kwargs).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    res_list = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )
    return res_list[0]  # {'boxes','scores','labels'}


def main():
    """
    Hyper parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="car. tire.")
    parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/grounded_sam2_hf_demo")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--topk", type=int, default=30, help="Top-K boxes to keep before SAM2 (to control VRAM)")
    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = args.text_prompt
    IMG_PATH = args.img_path
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_dump_json
    TOPK = max(1, args.topk)

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Enable TF32 if supported (Ampere+)
    if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build Grounding DINO
    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

    # setup the input image and text prompt
    text = TEXT_PROMPT  # must be lowercased + each label ends with a dot
    img_path = IMG_PATH

    image = Image.open(img_path)
    print(f"image: {image}")

    # SAM2 needs the image set once
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # autocast context for model forwards (bf16 if available, else fp16)
    use_bf16 = (DEVICE == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if DEVICE == "cuda" else torch.float32)

    # FIRST PASS GroundingDINO
    with torch.autocast(device_type=DEVICE, dtype=amp_dtype) if DEVICE == "cuda" else torch.no_grad():
        res = run_grounding(
            processor, grounding_model, DEVICE, image, text,
            box_thr=0.40, text_thr=0.30, size=None
        )

    # FALLBACK: larger working size + looser thresholds
    if len(res["boxes"]) == 0:
        with torch.autocast(device_type=DEVICE, dtype=amp_dtype) if DEVICE == "cuda" else torch.no_grad():
            res = run_grounding(
                processor, grounding_model, DEVICE, image, text,
                box_thr=0.25, text_thr=0.20,
                size={"shortest_edge": 1024, "longest_edge": 1536}
            )

    # Optional alt wording if still empty
    if len(res["boxes"]) == 0:
        alt_text = text + " vending machine. snack. bottle. can. logo. text."
        with torch.autocast(device_type=DEVICE, dtype=amp_dtype) if DEVICE == "cuda" else torch.no_grad():
            res = run_grounding(
                processor, grounding_model, DEVICE, image, alt_text,
                box_thr=0.25, text_thr=0.20,
                size={"shortest_edge": 1024, "longest_edge": 1536}
            )

    # If still no detections, exit gracefully
    if len(res["boxes"]) == 0:
        print("No detections after fallbacks; skipping SAM2.")
        img_np = cv2.imread(img_path)
        if img_np is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "no_detections.jpg"), img_np)
        return

    # Extract detections
    boxes = res["boxes"].cpu().numpy().astype(np.float32)      # (N, 4), xyxy
    scores_gdino = res["scores"].cpu().numpy()                 # (N,)
    labels_gdino = res["labels"]                                # list[str]
    N = boxes.shape[0]

    # Optional: keep Top-K by GroundingDINO score
    if N > TOPK:
        idx = np.argsort(-scores_gdino)[:TOPK]
        boxes = boxes[idx]
        scores_gdino = scores_gdino[idx]
        labels_gdino = [labels_gdino[i] for i in idx.tolist()]
        N = boxes.shape[0]

    # Prepare SAM2 prompt: one image, many boxes
    input_boxes = boxes[None, :, :]  # (1, N, 4)
    print("boxes shape before predict:", input_boxes.shape)

    # Run SAM2 (can also benefit from autocast)
    with torch.autocast(device_type=DEVICE, dtype=amp_dtype) if DEVICE == "cuda" else torch.no_grad():
        masks, sam_scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

     # If it's a torch tensor, move to CPU numpy
    if isinstance(sam_scores, torch.Tensor):
        sam_scores = sam_scores.detach().cpu().numpy()

    # Ensure numpy array, then squeeze batch/ singleton dims
    sam_scores = np.asarray(sam_scores).squeeze()

    # If SAM2 kept (1, N) shape, take the first row
    if sam_scores.ndim == 2 and sam_scores.shape[0] == 1:
        sam_scores = sam_scores[0]

    # Final list of floats, length == N (number of boxes/masks)
    sam_scores_list = [float(x) for x in sam_scores.reshape(-1)]
    assert len(sam_scores_list) == boxes.shape[0], \
        f"SAM2 scores len {len(sam_scores_list)} != boxes {boxes.shape[0]}"   


    # Normalize masks to (N, H, W)
    if masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    assert masks.ndim == 3 and masks.shape[0] == boxes.shape[0], \
        f"Mask/box mismatch: masks {masks.shape}, boxes {boxes.shape}"

    # Build labels for visualization (GroundingDINO confidences)
    confidences = scores_gdino.tolist()
    class_names = labels_gdino
    class_ids = np.arange(N, dtype=np.int32)

    labels_vis = [f"{cn} {cf:.2f}" for cn, cf in zip(class_names, confidences)]

    # Visualization with supervision
    img_bgr = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=boxes,                      # (N, 4)  <-- NOT (1, N, 4)
        mask=masks.astype(bool),         # (N, H, W)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_masks = mask_annotator.annotate(scene=annotated, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_masks)

    # JSON dump
    if DUMP_JSON_RESULTS:
        mask_rles = [single_mask_to_rle(m) for m in masks]

        results_json = {
            "image_path": img_path,
            "annotations": [
                {
                    "class_name": cls,
                    "bbox": box,
                    "segmentation": rle,
                    "score": sc,  # already float
                }
                for cls, box, rle, sc in zip(class_names, boxes.tolist(), mask_rles, sam_scores_list)
            ],
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }
        with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
            json.dump(results_json, f, indent=4)

if __name__ == "__main__":
    main()
