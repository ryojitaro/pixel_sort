import io

import cv2
import numpy as np

from rotate import derotate_image, rotate_image


color_space_dict = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV_FULL,
    "HLS": cv2.COLOR_BGR2HLS_FULL,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "Lab": cv2.COLOR_BGR2Lab,
    "Luv": cv2.COLOR_BGR2Luv,
    "XYZ": cv2.COLOR_BGR2XYZ,
}


def pixel_sort(
    file_bytes: io.BytesIO,
    color_mode: str,
    ch: int,
    threshold_mode: int,
    lower: int,
    upper: int,
    angle: int | float,
):
    img = cv2.imdecode(np.frombuffer(file_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    match threshold_mode:
        case 0:
            threshold_bool = True
            step = 2
        case 1:
            threshold_bool = False
            step = 2
        case 2:
            sort_start = 0
            step = 1

    angle = angle - 90
    h, w = img.shape[:2]
    img, mask = rotate_image(img, angle)
    color_space = cv2.cvtColor(img, color_space_dict[color_mode])
    mono = color_space[:, :, ch]
    # maskの範囲内で2値化する
    threshold = np.where(mask, (mono >= lower) & (mono <= upper), mask)

    for i, row in enumerate(mono):
        top_mask = mask[i].tolist().index(True)
        diff = np.diff(threshold[i, top_mask : top_mask + np.count_nonzero(mask[i])])
        where = np.where(diff)[0] + 1
        where = np.insert(where, 0, 0)
        where = np.append(where, np.count_nonzero(mask[i]))

        sort_start = (
            0
            if threshold_mode != 2 and threshold[i, mask[i]][0] == threshold_bool
            else 1
        )

        for j in range(sort_start, len(where) - 1, step):
            sort_range = row[top_mask + where[j] : top_mask + where[j + 1]]
            if len(sort_range) == 1:
                continue
            grp_sorted = np.argsort(sort_range)
            img[i, top_mask + where[j] : top_mask + where[j + 1]] = img[
                i, top_mask + where[j] : top_mask + where[j + 1]
            ][grp_sorted]

    img = derotate_image(img, -angle, h, w)
    return bytes(bytearray(cv2.imencode(".jpg", img)[1]))
