from numba import cuda
import math
import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace

@cuda.jit
def smooth_image(image):
    dimen = 10
    h, w, _ = image.shape
    py, px = cuda.grid(2)
    if py < h and px < w:
        value_sum_r = 0
        value_sum_g = 0
        value_sum_b = 0
        value_count = 0
        for y in range(py - dimen, py + dimen):
            for x in range(px - dimen, px + dimen):
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                value_sum_r = value_sum_r + image[y, x, 0]
                value_sum_g = value_sum_g + image[y, x, 1]
                value_sum_b = value_sum_b + image[y, x, 2]
                value_count = value_count + 1
        image[py, px, 0] = value_sum_r / value_count
        image[py, px, 1] = value_sum_g / value_count
        image[py, px, 2] = value_sum_b / value_count


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow('frame_original', frame)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(frame.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(frame.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        smooth_image[blockspergrid, threadsperblock](frame)
        cv2.imshow('frame_processed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
