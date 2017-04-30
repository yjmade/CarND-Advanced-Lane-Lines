# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2


class FrameQueue(object):

    def __init__(self, size):
        self.size = size
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)
        if len(self.queue) > self.size:
            self.queue.pop(0)

    def sum(self):
        return np.concatenate(self.queue, 0)


class LineFit(object):

    def __init__(self, is_video):
        if is_video:
            self.left_x_queue = FrameQueue(self.queue_size)
            self.left_y_queue = FrameQueue(self.queue_size)
            self.right_x_queue = FrameQueue(self.queue_size)
            self.right_y_queue = FrameQueue(self.queue_size)

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
            plt.axis('off')
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, self.width)
            plt.ylim(self.height, 0)
            fig.axes[0].get_yaxis().set_visible(False)
            fig.axes[0].get_xaxis().set_visible(False)
            # import ipdb; ipdb.set_trace()
            self.fig_write(fig, "poly_fit")

        return ploty, left_fitx, right_fitx

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def compute_curvature_and_root_point(self, y, x):
        a, b, c = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        y_eval = np.max(y) * self.ym_per_pix
        root_point = a * y_eval * y_eval + b * y_eval + c
        return (1 + (2 * a * y_eval + b)**2)**1.5 / abs(2 * a), root_point

    def compute_offset(self, root1, root2):
        middle_of_lane = (root1 + root2) / 2
        car_position = self.width / 2 * self.xm_per_pix
        return (car_position - middle_of_lane)

    def fit_line(self, img, plot):
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = self.extract_lanes_pixels(img)
        if self.is_video:
            self.left_x_queue.enqueue(leftx)
            leftx = self.left_x_queue.sum()
            self.left_y_queue.enqueue(lefty)
            lefty = self.left_y_queue.sum()
            self.right_x_queue.enqueue(rightx)
            rightx = self.right_x_queue.sum()
            self.right_y_queue.enqueue(righty)
            righty = self.right_y_queue.sum()
        ploty, left_fitx, right_fitx = self.poly_fit(
            leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds,
            img,
            plot=plot
        )
        return ploty, left_fitx, right_fitx
