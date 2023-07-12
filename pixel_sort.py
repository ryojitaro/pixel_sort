import io

import cv2
import numpy as np

from rotate import derotate_image, rotate_image


COLOR_SPACE_DICT = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV_FULL,
    "HLS": cv2.COLOR_BGR2HLS_FULL,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "Lab": cv2.COLOR_BGR2Lab,
    "Luv": cv2.COLOR_BGR2Luv,
    "XYZ": cv2.COLOR_BGR2XYZ,
}

SORT_TARGET_DICT = {
    0: {2},
    1: {1},
    2: {1, 2},
}


def main(
    file_bytes: io.BytesIO,
    mask,
    select_color_space: str,
    ch: int,
    select_sort_target: int,
    lower: int,
    upper: int,
    angle: float,
    ispolar: bool,
    polar_deg: float | None,
) -> bytes:
    img = cv2.imdecode(np.frombuffer(file_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    mask = np.full((img.shape[:2]), 255, np.uint8) if mask is None else mask

    color_space = COLOR_SPACE_DICT[select_color_space]
    sort_target = SORT_TARGET_DICT[select_sort_target]

    if ispolar:
        img = polar(
            img, mask, color_space, ch, sort_target, lower, upper, angle, polar_deg
        )
    else:
        img = prepos(img, mask, color_space, ch, sort_target, lower, upper, angle)

    return bytes(bytearray(cv2.imencode(".png", img)[1]))


def polar(
    img: cv2.Mat,
    mask: np.ndarray,
    color_space: str,
    ch: int,
    sort_target: int,
    lower: int,
    upper: int,
    angle: float,
    polar_deg: float | None,
) -> cv2.Mat:
    # pre
    img = expand(img)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    r = int(np.ceil(np.sqrt(h**2 + w**2) / 2))
    s = int(np.ceil(2 * np.pi * r))
    flags = cv2.WARP_POLAR_LINEAR
    img = cv2.warpPolar(img, (r * 2, s * 2), (w / 2, h / 2), r, flags)
    mask = cv2.warpPolar(mask, (r * 2, s * 2), (w / 2, h / 2), r, flags)
    ichido = s * 2 / 360
    img = np.roll(img, int(-ichido * polar_deg), 0)
    mask = np.roll(mask, int(-ichido * polar_deg), 0)

    img = prepos(
        img,
        mask,
        color_space,
        ch,
        sort_target,
        lower,
        upper,
        angle,
    )

    # pos
    img = np.roll(img, int(ichido * polar_deg), 0)
    img = cv2.warpPolar(img, (w, h), (w / 2, h / 2), r, flags + cv2.WARP_INVERSE_MAP)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = shrink(img)

    return img


def prepos(
    img: cv2.Mat,
    mask: np.ndarray,
    color_space: str,
    ch: int,
    sort_target: int,
    lower: int,
    upper: int,
    angle: float,
) -> cv2.Mat:
    angle = angle - 90
    # pre
    h_w = img.shape[:2]
    img, mask = rotate_image(angle, img, mask)
    color_space = cv2.cvtColor(img, color_space)
    mono = color_space[:, :, ch]

    img = pixel_sort(img, mask, mono, sort_target, lower, upper)

    # pos
    img = derotate_image(-angle, img, h_w)

    return img


def expand(img: cv2.Mat) -> cv2.Mat:
    h, w = img.shape[:2]
    new_w = w + 2
    new_h = h + 2
    M = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    new_img = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    return new_img


def shrink(img: cv2.Mat) -> cv2.Mat:
    h, w = img.shape[:2]
    new_w = w - 2
    new_h = h - 2
    M = np.array([[1, 0, -1], [0, 1, -1]], dtype=float)
    new_img = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    return new_img


def threshold(
    lower: int, upper: int, arr: np.ndarray, mask_arr: np.ndarray
) -> np.ndarray:
    th = (arr >= lower) & (arr <= upper)
    th = th.astype(np.uint8) + 1
    result = np.where(mask_arr, th, 0)
    return result


def get_split_indexes(th: np.ndarray) -> np.ndarray:
    bool_arr = np.full(len(th) + 1, True)
    bool_arr[1:-1] = th[1:] != th[:-1]
    result = bool_arr.nonzero()[0]
    return result


def isin(arr: np.ndarray, sort_target: set) -> np.ndarray:
    result = np.full(len(arr), False)
    for i in range(len(arr)):
        if arr[i] in sort_target:
            result[i] = True
    return result


def pixel_sort(
    img: cv2.Mat,
    mask: np.ndarray,
    mono: np.ndarray,
    sort_target: set,
    lower: int,
    upper: int,
) -> cv2.Mat:
    for i in range(img.shape[0]):
        th = threshold(lower, upper, mono[i], mask[i])
        split_idxs = get_split_indexes(th)
        split_bool = isin(th[split_idxs[:-1]], sort_target)
        lengths = np.ediff1d(split_idxs)

        for idx, lgt in zip(split_idxs[:-1][split_bool], lengths[split_bool]):
            if lgt == 1:
                continue
            sort_grp = mono[i, idx : idx + lgt]
            sorted_grp = np.argsort(sort_grp)
            img[i, idx : idx + lgt] = img[i, idx : idx + lgt][sorted_grp]

    return img
