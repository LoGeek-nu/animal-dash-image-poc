"""OpenCV-based preprocessing: drawing-frame detection -> perspective correction ->
background removal -> content crop -> square padding.

Design doc pipeline (see project notes):
  撮影画像 -> 台紙の枠だけ切り出す -> 明るさ・影を補正 ->
  「白に近い画素」を背景と判定 -> 背景のalphaを0にする ->
  絵の周囲だけcrop -> 透過PNG

台紙デザイン（アニマルダッシュの実際の台紙）には、描画エリアを囲む太い罫線の枠が
印刷されている。枠の外側（タイトル文字、スタッフ向け説明、机の陰など）は暗い要素が
枠自体と地続きに写り込むことがあり、枠の外側輪郭だけを見ると誤ってそれらと合体した
形を検出してしまう。そのため枠の外側ではなく、枠というリングの内側にできる「穴」
（＝描画エリアの紙面そのもの）を検出する。穴は枠の外に何が写っていようと影響を
受けない独立した領域なので、より安定して描画エリアだけを切り出せる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    warped_bgr: np.ndarray
    illum_corrected_bgr: np.ndarray
    mask: np.ndarray
    rgba: np.ndarray
    cropped_rgba: np.ndarray
    padded_rgba: np.ndarray
    corners: np.ndarray | None
    debug: dict = field(default_factory=dict)


def find_frame_corners(
    image_bgr: np.ndarray,
    dark_thresh: int = 120,
    min_area_ratio: float = 0.15,
) -> np.ndarray | None:
    """Locate the printed drawing-frame's interior and return its 4 inner corners.

    Thresholds for "dark ink" rather than running generic Canny edge detection,
    so stray edges from desk texture/background don't compete with the frame.
    The frame is a closed ring, so instead of taking its *outer* contour
    (which merges with whatever dark clutter touches it from outside — title
    text, staff instructions, desk shadow — into one wrong blob), this looks
    at the *hole* enclosed by the ring: the drawing area itself. That hole is
    unaffected by anything outside the ring, so it isolates the frame's
    interior reliably regardless of what else is dark in the photo.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # Bridge the frame's rounded corners / dashed anti-aliasing into one closed ring.
    kernel_size = max(int(min(image_bgr.shape[:2]) * 0.01), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_CCOMP yields 2-level hierarchy info so holes (contours with a parent)
    # can be told apart from the dark blobs that enclose them.
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    holes = [c for c, h in zip(contours, hierarchy[0]) if h[3] != -1]
    if not holes:
        return None
    holes.sort(key=cv2.contourArea, reverse=True)

    for hole in holes[:5]:
        area = cv2.contourArea(hole)
        if area < image_area * min_area_ratio:
            continue
        perimeter = cv2.arcLength(hole, True)
        # Larger epsilon than a sharp-corner rectangle needs, since the frame's
        # rounded corners otherwise approximate to >4 vertices.
        approx = cv2.approxPolyDP(hole, 0.04 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    # Fallback: rotated bounding rect of the largest sufficiently-big hole.
    largest = holes[0]
    if cv2.contourArea(largest) < image_area * min_area_ratio:
        return None
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect).astype(np.float32)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def warp_perspective(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the quadrilateral in `corners` to a fronto-parallel rectangle.

    Output width/height are derived from the corner distances themselves (the
    classic four-point-transform approach) rather than assumed from a fixed
    paper ratio, since the target is now the printed drawing frame rather than
    the whole A4 sheet.
    """
    tl, tr, br, bl = order_points(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


def strip_frame_border(image_bgr: np.ndarray, inset_ratio: float = 0.01) -> np.ndarray:
    """Crop inward slightly after warping, as a safety margin against the frame's ink.

    `find_frame_corners` now returns the frame's *inner* edge (see its docstring),
    so the warped image's boundary should already be paper, not border ink. This
    small inset only guards against threshold/blur jitter in that edge detection
    leaving a sliver of ink behind, which `build_ink_mask` would otherwise treat
    as "ink" and let survive `crop_to_content` as an unwanted colored ring.
    """
    h, w = image_bgr.shape[:2]
    dy, dx = int(h * inset_ratio), int(w * inset_ratio)
    if h - 2 * dy <= 0 or w - 2 * dx <= 0:
        return image_bgr
    return image_bgr[dy : h - dy, dx : w - dx]


def correct_illumination(image_bgr: np.ndarray, blur_ksize: int = 55) -> np.ndarray:
    """Flatten uneven lighting/shadow by dividing out a heavily-blurred background estimate."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    background = np.where(background == 0, 1, background).astype(np.float32)
    normalized = (gray.astype(np.float32) / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    result = image_bgr.copy()
    gray_safe = np.where(gray == 0, 1, gray).astype(np.float32)
    scale = (normalized.astype(np.float32) / gray_safe)[..., None]
    result = np.clip(image_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return result


def build_ink_mask(
    image_bgr: np.ndarray,
    brightness_thresh: int = 225,
    saturation_thresh: int = 30,
    gap_close_ratio: float = 0.01,
) -> np.ndarray:
    """Mask of "ink" pixels: everything except paper background reachable from the edges.

    A plain brightness/saturation threshold would also erase light-colored fills
    *inside* the character outline (e.g. a white-bellied dinosaur, or an
    almost-white cat drawn with a sketchy/crayon-textured outline) since they
    look just as "background-like" as the surrounding paper. So only
    near-white/low-saturation pixels connected (8-connectivity) to the image
    border count as background.

    Sketchy/crayon-style outlines are often drawn with tiny gaps (dashed,
    textured strokes rather than one solid line), which would otherwise let
    the outside background "leak" through the outline into enclosed light
    fills and wrongly erase them too. So the non-background ("barrier") side
    is morphologically closed first, sealing those gaps, before checking
    border-connectivity.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]

    is_background_candidate = (gray > brightness_thresh) & (saturation < saturation_thresh)
    candidate_mask = np.where(is_background_candidate, 255, 0).astype(np.uint8)

    barrier_mask = 255 - candidate_mask
    gap_close_ksize = max(int(min(image_bgr.shape[:2]) * gap_close_ratio), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_close_ksize, gap_close_ksize))
    barrier_sealed = cv2.morphologyEx(barrier_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    sealed_candidate = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(barrier_sealed))

    _, labels = cv2.connectedComponents(sealed_candidate, connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)  # label 0 = non-candidate (ink/barrier) pixels, not background

    background = np.isin(labels, list(border_labels)).astype(np.uint8) * 255
    mask = 255 - background

    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask


def apply_mask_alpha(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def crop_to_content(image_bgra: np.ndarray, margin: int = 20) -> np.ndarray:
    alpha = image_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return image_bgra

    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, image_bgra.shape[1])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, image_bgra.shape[0])
    return image_bgra[y0:y1, x0:x1]


def pad_to_square(image_bgra: np.ndarray, size: int | None = None) -> np.ndarray:
    h, w = image_bgra.shape[:2]
    side = size or max(h, w)

    scale = min(side / h, side / w, 1.0) if max(h, w) > side else 1.0
    if scale != 1.0:
        image_bgra = cv2.resize(image_bgra, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image_bgra.shape[:2]

    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = image_bgra
    return canvas


def process_image(
    image_bgr: np.ndarray,
    output_square_size: int = 800,
    crop_margin: int = 20,
) -> PreprocessResult:
    corners = find_frame_corners(image_bgr)
    warped = warp_perspective(image_bgr, corners) if corners is not None else image_bgr.copy()
    warped = strip_frame_border(warped) if corners is not None else warped

    illum_corrected = correct_illumination(warped)
    mask = build_ink_mask(illum_corrected)
    rgba = apply_mask_alpha(illum_corrected, mask)
    cropped = crop_to_content(rgba, margin=crop_margin)
    padded = pad_to_square(cropped, size=output_square_size)

    return PreprocessResult(
        warped_bgr=warped,
        illum_corrected_bgr=illum_corrected,
        mask=mask,
        rgba=rgba,
        cropped_rgba=cropped,
        padded_rgba=padded,
        corners=corners,
    )
