import cv2
import numpy as np


def rotate_image(img, angle) -> tuple[cv2.Mat, np.ndarray]:
    h, w = img.shape[:2]
    rad = np.radians(angle)
    # 回転角度のsinとcos
    cos_rad = abs(np.cos(rad))
    sin_rad = abs(np.sin(rad))
    # 角度が0ならw*1 + h*0でそのまま
    w_rot = int(np.round(w * cos_rad + h * sin_rad))
    h_rot = int(np.round(w * sin_rad + h * cos_rad))

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 移動量
    matrix[0][2] += w_rot / 2 - w / 2
    matrix[1][2] += h_rot / 2 - h / 2

    img_rot = cv2.warpAffine(
        img, matrix, (w_rot, h_rot), borderMode=cv2.BORDER_REPLICATE
    )
    mask = np.full((*img.shape[:2], 1), 255, np.uint8)
    mask = cv2.warpAffine(mask, matrix, (w_rot, h_rot), borderValue=0)
    mask = mask.astype(np.bool_)
    return img_rot, mask


def derotate_image(img, angle, orig_h, orig_w) -> cv2.Mat:
    h, w = img.shape[:2]
    rad = np.radians(angle)
    # 回転角度のsinとcos
    cos_rad = abs(np.cos(rad))
    sin_rad = abs(np.sin(rad))

    w_rot = int(np.round(orig_w * cos_rad + orig_h * sin_rad))
    h_rot = int(np.round(orig_w * sin_rad + orig_h * cos_rad))

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 移動量
    matrix[0][2] += orig_w / 2 + w_rot / -2
    matrix[1][2] += orig_h / 2 + h_rot / -2

    img_rot = cv2.warpAffine(img, matrix, (orig_w, orig_h))

    return img_rot
