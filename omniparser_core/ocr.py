from typing import Union

import cv2
import easyocr
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR

from omniparser_core.rendering import get_xywh, get_xyxy

reader = easyocr.Reader(["en"])

paddle_ocr = PaddleOCR(
        lang="en",
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

def _as_xyxy_quad(points):
    # Normalize to 4-point quad for existing coordinate conversion helpers.
    if points is None:
        return None
    arr = np.asarray(points).reshape(-1, 2)
    if arr.shape[0] < 4:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _extract_from_paddle_predict(prediction, text_threshold):
    # PaddleOCR 3.x predict output commonly has rec_text/rec_score/dt_polys.
    if hasattr(prediction, "res"):
        prediction = prediction.res

    if isinstance(prediction, dict):
        texts = prediction.get("rec_text") or prediction.get("texts") or []
        scores = prediction.get("rec_score") or prediction.get("scores") or []
        polys = prediction.get("dt_polys") or prediction.get("polys") or []
        text_out, coord_out = [], []
        for txt, score, poly in zip(texts, scores, polys):
            if score is None or score >= text_threshold:
                quad = _as_xyxy_quad(poly)
                if quad is not None:
                    text_out.append(txt)
                    coord_out.append(quad)
        return text_out, coord_out

    return [], []


def _run_paddle_ocr(image_np, text_threshold):
    try:
        predictions = paddle_ocr.predict(image_np)
        text, coord = [], []
        if predictions is not None:
            for pred in predictions:
                t, c = _extract_from_paddle_predict(pred, text_threshold)
                text.extend(t)
                coord.extend(c)
        return text, coord
    except Exception as e:
        print(f"PaddleOCR failed ({type(e).__name__}): {e}. Falling back to EasyOCR.")
        result = reader.readtext(image_np)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
        return text, coord


def check_ocr_box(
    image_source: Union[str, Image.Image],
    display_img=True,
    output_bb_format="xywh",
    goal_filtering=None,
    easyocr_args=None,
    use_paddleocr=False,
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == "RGBA":
        image_source = image_source.convert("RGB")
    image_np = np.array(image_source)

    if use_paddleocr:
        text_threshold = 0.5 if easyocr_args is None else easyocr_args["text_threshold"]
        text, coord = _run_paddle_ocr(image_np, text_threshold)
    else:
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]

    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x + a, y + b), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == "xywh":
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == "xyxy":
            bb = [get_xyxy(item) for item in coord]
        else:
            raise ValueError(f"Unsupported output_bb_format: {output_bb_format}")
    return (text, bb), goal_filtering
