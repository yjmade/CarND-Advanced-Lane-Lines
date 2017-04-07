# -*- coding: utf-8 -*-
import numpy as np
import cv2

from calibrate import _undistort as undistort


def sobel_thresh(
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
    result = np.ones_like(gray)

    if abs_thresh_x:
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= abs_thresh_x[0]) & (scaled_sobel <= abs_thresh_x[1])] = 1
        result &= binary_output

    if abs_thresh_y:
        scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        binary_output = np.zeros_like(scaled_sobel)
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
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        result &= binary_output

    if dir_thresh:
        absgraddir = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
        result &= binary_output

    # Return the binary image
    return result


def hls_thresh(img, thresh):  # (170, 255) or (90, 255)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    binary_output = np.zeros_like(hls)
    s = hls[:, :, 2]
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary_output

src = [
    (),
    (),
    (),
    ()
]

dst = [
    (),
    (),
    (),
    ()
]

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)


def perspective_transform(img):
    return cv2.warpPerspective(img, M, img.shape[:2], flags=cv2.INTER_LINEAR)


def inv_perspective_transform(img):
    return cv2.warpPerspective(img, M_inv, img.shape[:2], flags=cv2.INTER_LINEAR)


def main(img):
    img = undistort(img)
    mask = sobel_thresh(
        img,
        sobel_kernel=3,
        abs_thresh_x=None,
        abs_thresh_y=None,
        mag_thresh=None,
        dir_thresh=None
    )
    mask &= hls_thresh(img, thresh=[])

    warped_img = perspective_transform(img)
