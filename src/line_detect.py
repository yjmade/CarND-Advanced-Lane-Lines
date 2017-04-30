# -*- coding: utf-8 -*-
import os
import sys
import math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from calibrate import _undistort as undistort
import click
from moviepy.editor import VideoFileClip
from line_fit import LineFit

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class LineDetect(LineFit):

    result_dir = "./output_images"

    def _get_imwrite_name(self, name):
        name = os.path.join(self.result_dir, self.fname % (self.imwrite_index, name))
        self.imwrite_index += 1
        return name

    def imwrite(self, img, name):
        if self.concat_draw is not None:
            # img = cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.names.append(name)
            self.concat_draw.append(img)
        if self.debug:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._get_imwrite_name(name), img)

    # def fig_write(self, fig, name):
    #     from io import BytesIO
    #     io = BytesIO()
    #     fig.savefig(io, dpi=300, bbox_inches='tight', pad_inches=0)
    #     io.seek(0)
    #     img = cv2.cvtColor(
    #         cv2.imdecode(np.fromstring(io.read(), np.uint8), cv2.IMREAD_COLOR),
    #         cv2.COLOR_BGR2RGB
    #     )
    #     self.imwrite(img, name)

    def fig_write(self, fig, name):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        res_str, shape = canvas.print_to_buffer()
        img = np.fromstring(res_str, dtype='uint8').reshape([shape[1], shape[0], 4])
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # import ipdb; ipdb.set_trace()
        self.imwrite(img, name)

    def normalize_img_to_3_channel(self, img):
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        return img

    def summarize_concat_draw(self, img_per_row=4):
        concat_draw = [
            self.normalize_img_to_3_channel(img)
            for img in self.concat_draw
        ]
        main_res = concat_draw.pop()
        concat_draw = [
            cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            for img, name in zip(concat_draw, self.names)
        ]
        row = math.ceil(len(concat_draw) / img_per_row)
        sub_height = self.height // img_per_row
        sub_width = self.width // img_per_row

        row_offset = sub_height * row
        vis = np.zeros([row_offset + self.height, self.width, 3], dtype=np.uint8)
        vis[row_offset:self.height + row_offset, :, :] = main_res
        for i in range(row):
            for j in range(img_per_row):
                try:
                    img = concat_draw.pop(0)
                except IndexError:
                    break
                height_start = i * sub_height
                height_stop = height_start + sub_height
                width_start = j * sub_width
                width_stop = width_start + sub_width
                vis[height_start:height_stop, width_start:width_stop, :] = cv2.resize(img, (sub_width, sub_height))

        return vis

    def imwrite_mask(self, mask, name):
        self.imwrite(mask * 255, name)

    def __init__(self, is_video=False, debug=True, concat_draw=False, fpath=None):
        self.debug = debug
        self.concat_draw = True if concat_draw else None
        self.is_video = is_video
        self.seq = 0
        self.fpath = fpath
        super(LineDetect, self).__init__(self.is_video)

    def init_main(self, img_or_fname):
        self.seq += 1
        if self.concat_draw:
            self.concat_draw = []
            self.names = []
        if isinstance(img_or_fname, str):
            img = cv2.imread(img_or_fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fname = os.path.split(img_or_fname)[-1]
            fname = "{}-%d-%s{}".format(*os.path.splitext(fname))

        elif isinstance(img_or_fname, np.ndarray):
            img = img_or_fname
            fname = self.fpath
            if self.is_video:
                assert fname
                sp = fname.split("/")
                sp[-1] = "{}-{}-%d-%s.png".format(self.seq, os.path.splitext(sp[-1])[0])
                fname = "/".join(sp)
        else:
            raise ValueError("input is neither a filename or a img array")

        if self.debug:
            assert fname
        self.fname = fname
        self.height, self.width = img.shape[:2]
        self.imwrite_index = 0

        return img

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
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls_img[:, :, 2]
        self.imwrite(s_channel, "s_channel")
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    @property
    def perspective_transform_src_dst(self):
        return (
            np.array([
                [565, 470],
                [210, self.height],
                [1120, self.height],
                [715, 465]
            ], dtype=np.float32),
            np.array([
                [210, 200],
                [210, self.height],
                [1120, self.height],
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

    def final_draw(self, undist_img, ploty, left_fitx, right_fitx, curvature, offset):
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
        cv2.putText(result, "Curvature: %.2fm Offset: %.2fm" % (curvature, offset), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.imwrite(result, "result")
        if self.concat_draw is not None:
            result = self.summarize_concat_draw()
            self.imwrite(result, "sumresult")

        return result

    def main(self, img):
        img = self.init_main(img)
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
        ploty, left_fitx, right_fitx = self.fit_line(warped_mask, self.debug or self.concat_draw is not None)
        left_curvature, left_root = self.compute_curvature_and_root_point(ploty, left_fitx)
        right_curvature, right_root = self.compute_curvature_and_root_point(ploty, right_fitx)
        mean_curvature = (left_curvature + right_curvature) / 2
        offset = self.compute_offset(left_root, right_root)
        if self.debug:
            print(self.fname)
            print("left curvature", left_curvature)
            print("right curvature", right_curvature)
            print("mean curvature", mean_curvature)
            print("offset", offset)
        return self.final_draw(img, ploty, left_fitx, right_fitx, mean_curvature, offset)


@click.command()
@click.argument("input_video_path", type=click.Path(exists=True))
@click.argument("output_path")
@click.option("--video_debug", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--subclip")
def main(input_video_path, output_path, video_debug, debug, subclip=None):
    detector = LineDetect(debug=debug, concat_draw=video_debug, is_video=True, fpath=output_path)
    clip = VideoFileClip(input_video_path)
    if subclip:
        subclip = eval("(%s)" % subclip)
        clip = clip.subclip(*subclip)
    clip = clip.fl_image(lambda img: detector.main(img))
    clip.write_videofile(output_path, audio=False, threads=8)

if __name__ == "__main__":
    main()
# python src/line_detect.py project_video.mp4 output_images/project_video.mp4
