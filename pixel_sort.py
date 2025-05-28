import io
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import cv2
import numpy as np
from cv2.typing import MatLike

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

SORT_TARGETS_DICT = {
    "in": (2,),
    "out": (1,),
    "both": (1, 2),
}


@dataclass
class PixelsortConfig:
    color_space: str
    ch: int
    sort_targets: str
    lower: int
    upper: int
    angle: float
    ispolar: bool
    polar_deg: float


class PixelSort:
    def __init__(
        self,
        img_file: io.BytesIO,
        cfg: PixelsortConfig,
    ):
        self.org_img = cv2.imdecode(
            np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR
        )
        self.img = self.org_img.copy()
        self.mask: MatLike = np.full((self.org_img.shape[:2]), 255, np.uint8)
        self.cfg = cfg

    @contextmanager
    def polar_ctx(self):
        self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        self.mask = cv2.rotate(self.mask, cv2.ROTATE_90_CLOCKWISE)
        h, w = self.img.shape[:2]
        radial_res = int(np.ceil(np.hypot(h, w) / 2))
        angular_res = int(np.ceil(2 * np.pi * radial_res))
        dims = (2 * radial_res, 2 * angular_res)
        center = (w / 2, h / 2)
        self.img = cv2.warpPolar(
            self.img,
            dims,
            center,
            radial_res,
            cv2.WARP_POLAR_LINEAR,
        )
        self.mask = cv2.warpPolar(
            self.mask,
            dims,
            center,
            radial_res,
            cv2.WARP_POLAR_LINEAR,
        )
        step = angular_res * 2 / 360
        self.img = np.roll(self.img, int(-step * self.cfg.polar_deg), 0)
        self.mask = np.roll(self.mask, int(-step * self.cfg.polar_deg), 0)

        yield

        self.img = np.roll(self.img, int(step * self.cfg.polar_deg), 0)
        self.img = cv2.warpPolar(
            self.img,
            (w, h),
            center,
            radial_res,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @contextmanager
    def prepost_ctx(self):
        angle = self.cfg.angle - 90
        h_w: tuple[int, int] = self.img.shape[:2]
        self.img, self.mask = rotate_image(self.img, self.mask, angle)
        convert_color_space_img = cv2.cvtColor(
            self.img, COLOR_SPACE_DICT[self.cfg.color_space]
        )
        mono = convert_color_space_img[:, :, self.cfg.ch]
        yield mono
        self.img = derotate_image(self.img, -angle, h_w)

    def separation(self, mono: MatLike):
        targets = SORT_TARGETS_DICT[self.cfg.sort_targets]
        sep = ((self.cfg.lower <= mono) & (mono <= self.cfg.upper)).astype(np.uint8)
        if len(targets) == 1:
            if targets[0] == 1:
                sep = 1 - sep
        else:
            sep += 1
        sep *= 0 < self.mask
        return sep

    def pixel_sort(self, mono: MatLike):
        sep = self.separation(mono)
        for i in range(mono.shape[0]):
            change_indices = np.where(np.ediff1d(sep[i]) != 0)[0] + 1
            boundaries = np.r_[0, change_indices, len(sep[i])]
            starts = boundaries[:-1]
            ends = boundaries[1:]
            mask = np.isin(sep[i][starts], (1, 2))

            for start, end in zip(starts[mask], ends[mask]):
                sorted_grp = np.argsort(mono[i, start:end])
                self.img[i, start:end] = self.img[i, start:end][sorted_grp]

    def main(self) -> bytes:
        self.img = self.org_img

        with ExitStack() as stack:
            if self.cfg.ispolar:
                stack.enter_context(self.polar_ctx())
            mono = stack.enter_context(self.prepost_ctx())
            self.pixel_sort(mono)

        return bytes(bytearray(cv2.imencode(".png", self.img)[1]))
