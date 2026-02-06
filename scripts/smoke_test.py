#!/usr/bin/env python3
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

from omniparser_core import (
    ScreenParserPipeline,
    ScreenParserPipelineConfig,
    get_caption_model,
    get_yolo_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OmniParser smoke test")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input screenshot image",
    )
    parser.add_argument(
        "--som-model",
        type=str,
        default="weights/icon_detect/model.pt",
        help="Path to icon detection model (.pt)",
    )
    parser.add_argument(
        "--caption-model-name",
        type=str,
        default="florence2",
        help="Caption model name (e.g. florence2, blip2)",
    )
    parser.add_argument(
        "--caption-model-path",
        type=str,
        default="weights/icon_caption_florence",
        help="Path to caption model directory",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.05,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Overlap filtering IoU threshold",
    )
    parser.add_argument(
        "--use-paddleocr",
        action="store_true",
        help="Use PaddleOCR instead of EasyOCR",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Icon caption batch size",
    )
    parser.add_argument(
        "--save-annotated",
        type=str,
        default="",
        help="Optional output path for annotated image PNG",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for parsed content + metrics JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    yolo_model = get_yolo_model(args.som_model)
    caption_model = get_caption_model(args.caption_model_name, args.caption_model_path)
    pipeline = ScreenParserPipeline(
        yolo_model=yolo_model,
        caption_model=caption_model,
        config=ScreenParserPipelineConfig(
            box_threshold=args.box_threshold,
            use_paddleocr=args.use_paddleocr,
            iou_threshold=args.iou_threshold,
            batch_size=args.batch_size,
        ),
    )

    som_img_b64, label_coordinates, parsed_content_list, metrics = pipeline.parse_image(image)

    print(f"image: {image_path}")
    print(f"parsed_elements: {len(parsed_content_list)}")
    print(
        "metrics_ms:",
        json.dumps(
            {
                "ocr_ms": round(metrics.ocr_ms, 2),
                "icon_detect_ms": round(metrics.icon_detect_ms, 2),
                "build_elements_ms": round(metrics.build_elements_ms, 2),
                "icon_caption_ms": round(metrics.icon_caption_ms, 2),
                "render_ms": round(metrics.render_ms, 2),
                "total_ms": round(metrics.total_ms, 2),
            }
        ),
    )

    if args.save_annotated:
        annotated = Image.open(io.BytesIO(base64.b64decode(som_img_b64)))
        out_path = Path(args.save_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        print(f"saved_annotated: {out_path}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_coordinates": label_coordinates,
            "parsed_content_list": parsed_content_list,
            "metrics_ms": {
                "ocr_ms": metrics.ocr_ms,
                "icon_detect_ms": metrics.icon_detect_ms,
                "build_elements_ms": metrics.build_elements_ms,
                "icon_caption_ms": metrics.icon_caption_ms,
                "render_ms": metrics.render_ms,
                "total_ms": metrics.total_ms,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"saved_json: {out_path}")


if __name__ == "__main__":
    main()
