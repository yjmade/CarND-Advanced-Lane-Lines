# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2

from .calibrate import _undistort as undistort


class LineDetect(object):

    result_dir = "./output_images"

    def imwrite(self, img, name):
        if not self.debug:
            return
        cv2.imwrite(os.path.join(self.result_dir, self.fname % (self.imwrite_index, name)), img)
        self.imwrite_index += 1

    def imwrite_mask(self, mask, name):
        if not self.debug:
            return
        self.imwrite(mask * 255, name)

    def __init__(self, img_or_fname, debug=None, fname=None):
        if isinstance(img_or_fname, str):
            img = cv2.imread(img_or_fname)
            debug = True if debug is None else debug
            fname = os.path.split(img_or_fname)[-1]
            fname = "{}-%d-%s{}".format(*os.path.splitext(fname))

        elif isinstance(img_or_fname, np.array):
            img = img_or_fname
            debug = False if debug is None else debug
        else:
            raise ValueError("input is neither a filename or a img array")

        if debug:
            assert fname
        self.img = img
        self.fname = fname
        self.debug = debug

    def binary_thresh(
        self,
        img,
        sobel_kernel=3,
        abs_thresh_x=None,  # (0, 255)
        abs_thresh_y=None,  # (0, 255)
        mag_thresh=None,  # (0, 255)
        dir_thresh=None  # (0, np.pi / 2)
    ):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        result = np.ones(gray.shape, dtype=np.uint8)
        if abs_thresh_x:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
            binary_output = np.zeros_like(result)
            # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
            binary_output[(scaled_sobel >= abs_thresh_x[0]) & (scaled_sobel <= abs_thresh_x[1])] = 1
            result &= binary_output

        if abs_thresh_y:
            scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
            binary_output = np.zeros_like(result)
            # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
            binary_output[(scaled_sobel >= abs_thresh_y[0]) & (scaled_sobel <= abs_thresh_y[1])] = 1
            result &= binary_output

        if mag_thresh:
            # Calculate the gradient magnitude
            gradmag = np.sqrt(sobelx**2 + sobely**2)
            # Rescale to 8 bit
            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            # Create a binary image of ones where threshold is met, zeros otherwise
            binary_output = np.zeros_like(result)
            binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
            result &= binary_output

        if dir_thresh:
            absgraddir = np.arctan2(abs_sobely, abs_sobelx)
            binary_output = np.zeros_like(result)
            binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
            result &= binary_output

        # Return the binary image
        return result

    def hls_thresh(self, img, thresh):  # (170, 255) or (90, 255)
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls_img[:, :, 2]
        self.imwrite(s_channel, "s_channel")
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    @property
    def perspective_transform_src_dst(self):
        img_height, img_width = self.img.shape[:2]
        return (
            np.array([
                [565, 470],
                [210, img_height],
                [1120, img_height],
                [715, 465]
            ], dtype=np.float32),
            np.array([
                [210, 200],
                [210, img_height],
                [1120, img_height],
                [1120, 200]
            ], dtype=np.float32),
        )

    @property
    def perspective_transform_m(self):
        src, dst = self.perspective_transform_src_dst
        return cv2.getPerspectiveTransform(src, dst)

    @property
    def perspective_transform_m_inv(self):
        src, dst = self.perspective_transform_src_dst
        return cv2.getPerspectiveTransform(dst, src)

    def perspective_transform(self, img):
        return cv2.warpPerspective(img, self.perspective_transform_m, img.shape[2::-1], flags=cv2.INTER_LINEAR)

    def inv_perspective_transform(self, img):
        return cv2.warpPerspective(img, self.perspective_transform_m_inv, img.shape[2::-1], flags=cv2.INTER_LINEAR)

    def main(self):
        self.imwrite_index = 0
        img = self.img
        self.imwrite(img, "orig")
        img = undistort(img)
        self.imwrite(img, "undistort")
        mask = self.binary_thresh(
            img,
            sobel_kernel=3,
            abs_thresh_x=(20, 100),
            abs_thresh_y=None,
            mag_thresh=None,
            dir_thresh=(0.7, 1.3)
        )
        self.imwrite_mask(mask, "binary_thresh")
        hls_mask = self.hls_thresh(img, thresh=[150, 255])
        self.imwrite_mask(hls_mask, "hls_thresh")
        mask |= hls_mask
        self.imwrite_mask(mask, "binary_thresh_with_hls_thresh")

        warped_mask = self.perspective_transform(mask)
        self.imwrite_mask(warped_mask, "perspective_transformed")
