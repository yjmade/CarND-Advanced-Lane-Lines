# -*- coding: utf-8 -*-
import pickle
import cv2
import os
import numpy as np
import click
import glob


@click.command()
@click.argument("image_folder_path")
@click.option("--nx", default=9)
@click.option("--ny", default=6)
@click.option("--demo", default=True)
def calibrate(image_folder_path, nx=9, ny=6, demo=False):
    raw_imgs = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in glob.iglob(os.path.join(image_folder_path, "*.jpg"))]
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    imgpoints = []
    objpoints = []
    for i, img in enumerate(raw_imgs):
        ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
        # print(corners)
        # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # plt.imsave("results/%d.jpg" % i, img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    with open("calibrate.pickle", mode="wb") as f:
        pickle.dump({"mtx": mtx, "dist": dist}, f)

    if demo:
        # img = random.choice(raw_imgs)
        img = raw_imgs[0]
        undistort_img = cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)
        cv2.imshow("undistort", undistort_img)
        cv2.waitKey()


UNDISTORT_INFO = None


def _undistort(img):
    global UNDISTORT_INFO
    if UNDISTORT_INFO is None:
        with open("calibrate.pickle", "rb") as f:
            UNDISTORT_INFO = pickle.load(f)
    mtx = UNDISTORT_INFO["mtx"]
    dist = UNDISTORT_INFO["dist"]
    return cv2.undistort(img, mtx, dist, dst=None, newCameraMatrix=mtx)


@click.command()
@click.argument("image_folder_path")
def undistort(image_folder_path):
    for img_name in glob.iglob(os.path.join(image_folder_path, "*.jpg")):
        img = _undistort(cv2.imread(img_name))
        cv2.imwrite("output_images/%s" % os.path.split(img_name)[-1], img)


if __name__ == "__main__":
    @click.group()
    def cli():
        pass

    cli.add_command(calibrate)
    cli.add_command(undistort)

    cli()
