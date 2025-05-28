import cv2
import numpy as np
from cv2.typing import MatLike


def rotate_image(
    img: MatLike, mask: MatLike, angle: float
) -> tuple[MatLike, np.ndarray]:
    h, w = img.shape[:2]
    rad = np.radians(angle)
    # 回転角度のsinとcos
    cos_rad = abs(np.cos(rad))
    sin_rad = abs(np.sin(rad))
    # 角度が0ならw*1 + h*0でそのまま
    w_rot = int(np.round(w * cos_rad + h * sin_rad))
    h_rot = int(np.round(w * sin_rad + h * cos_rad))

    matrix = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1)
    # 移動量
    matrix[0][2] += w_rot / 2 - w / 2
    matrix[1][2] += h_rot / 2 - h / 2

    img_rot = cv2.warpAffine(
        img, matrix, (w_rot, h_rot), borderMode=cv2.BORDER_REPLICATE
    )
    mask = cv2.warpAffine(mask, matrix, (w_rot, h_rot), borderMode=cv2.BORDER_REPLICATE)

    return img_rot, mask


def derotate_image(img: MatLike, angle: float, h_w: tuple[int, int]) -> MatLike:
    h, w = img.shape[:2]
    orig_h, orig_w = h_w
    rad = np.radians(angle)
    # 回転角度のsinとcos
    cos_rad = abs(np.cos(rad))
    sin_rad = abs(np.sin(rad))
    # 角度が0ならw*1 + h*0でそのまま
    w_rot = int(np.round(orig_w * cos_rad + orig_h * sin_rad))
    h_rot = int(np.round(orig_w * sin_rad + orig_h * cos_rad))

    matrix = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1)
    # 移動量
    matrix[0][2] += orig_w / 2 - w_rot / 2
    matrix[1][2] += orig_h / 2 - h_rot / 2

    img_rot = cv2.warpAffine(img, matrix, (orig_w, orig_h))

    return img_rot
