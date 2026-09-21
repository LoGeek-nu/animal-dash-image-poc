"""End-to-end orchestration: photo -> preprocessed transparent PNG -> Gemini status JSON."""

from __future__ import annotations

import json
import os

import cv2

from . import gemini_status
from .preprocessing import PreprocessResult, process_image


def save_debug_artifacts(result: PreprocessResult, debug_dir: str) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "01_warped.png"), result.warped_bgr)
    cv2.imwrite(os.path.join(debug_dir, "02_illum_corrected.png"), result.illum_corrected_bgr)
    cv2.imwrite(os.path.join(debug_dir, "03_mask.png"), result.mask)
    cv2.imwrite(os.path.join(debug_dir, "04_rgba.png"), result.rgba)
    cv2.imwrite(os.path.join(debug_dir, "05_cropped.png"), result.cropped_rgba)
    cv2.imwrite(os.path.join(debug_dir, "06_padded.png"), result.padded_rgba)


def preprocess_file(input_path: str, output_path: str, debug_dir: str | None = None) -> PreprocessResult:
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = process_image(image_bgr)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result.padded_rgba)

    if debug_dir:
        save_debug_artifacts(result, debug_dir)

    return result


def run_full_pipeline(
    input_path: str,
    output_prefix: str,
    api_key: str | None = None,
    debug_dir: str | None = None,
) -> dict:
    png_path = f"{output_prefix}.png"
    json_path = f"{output_prefix}.json"

    preprocess_file(input_path, png_path, debug_dir=debug_dir)
    status = gemini_status.generate_status(png_path, api_key=api_key)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    return status
