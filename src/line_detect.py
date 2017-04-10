# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
import matplotlib.pyplot as plt
from calibrate import _undistort as undistort
import click
from moviepy.editor import VideoFileClip

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class LineDetect(object):

    result_dir = "./output_images"

    def _get_imwrite_name(self, name):
        name = os.path.join(self.result_dir, self.fname % (self.imwrite_index, name))
        self.imwrite_index += 1
        return name

    def imwrite(self, img, name):
        if not self.debug:
            return
        cv2.imwrite(self._get_imwrite_name(name), img)

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

        elif isinstance(img_or_fname, np.ndarray):
            img = img_or_fname
            debug = False if debug is None else debug
        else:
            raise ValueError("input is neither a filename or a img array")

        if debug:
            assert fname
        self.img = img
        self.fname = fname
        self.debug = debug
        self.height, self.width = self.img.shape[:2]

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
        # self.imwrite(gray, "gray")
        # gray = clahe.apply(gray)
        # self.imwrite(gray, "equalizeHist")
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
        return cv2.warpPerspective(img, self.perspective_transform_m, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def inv_perspective_transform(self, img):
        return cv2.warpPerspective(img, self.perspective_transform_m_inv, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def extract_lanes_pixels(self, binary_warped):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)
        # plt.show()
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds

    def poly_fit(self, leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, binary_warped, plot=False):

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        if plot:
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            fig = plt.figure()
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, self.img.shape[1])
            plt.ylim(self.img.shape[0], 0)
            fig.savefig(self._get_imwrite_name("poly_fit"), dpi=300, bbox_inches='tight')

        return ploty, left_fitx, right_fitx

    def compute_curvature(self, y, x):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        a, b, c = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

        return (1 + (2 * a * self.height + b)**2)**1.5 / abs(2 * a)

    def final_draw(self, undist_img, ploty, left_fitx, right_fitx, curvature):
        # Create an image to draw the lines on
        warp_zero = np.zeros([self.height, self.width]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inv_perspective_transform(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
        cv2.putText(result, "Curvature: %.2f" % curvature, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.imwrite(result, "result")
        return result

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

        ploty, left_fitx, right_fitx = self.poly_fit(
            *self.extract_lanes_pixels(warped_mask),
            warped_mask,
            plot=self.debug
        )
        left_curvature = self.compute_curvature(ploty, left_fitx)
        right_curvature = self.compute_curvature(ploty, right_fitx)
        mean_curvature = (left_curvature + right_curvature) / 2
        if self.debug:
            print(self.fname)
            print("left curvature", left_curvature)
            print("right curvature", right_curvature)
            print("mean curvature", mean_curvature)
        return self.final_draw(img, ploty, left_fitx, right_fitx, mean_curvature)


@click.command()
@click.argument("input_video_path", type=click.Path(exists=True))
@click.argument("output_path")
def main(input_video_path, output_path):
    clip = VideoFileClip(input_video_path).fl_image(lambda img: LineDetect(img).main())
    clip.write_videofile(output_path, audio=False, threads=8)

if __name__ == "__main__":
    main()
# python src/line_detect.py project_video.mp4 output_images/project_video.mp4
