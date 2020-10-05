import numpy as np
import cv2
from skimage import draw


# 用法： my_hog = Hog((image_height, image_width), block_size in cell, block_stride in cell, cell_size in pixel,
# full-angle, nbin)
# 例如：cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9) <=> Hog((128, 64), (2, 2), (1, 1), (8, 8))
# v = my_hog.compute(img) 即得到HOG向量

class Hog:

    def __init__(self, window_size, block_size, block_stride, cell_size, full_angle=180, nbin=9):
        assert len(window_size) == 2
        assert len(block_size) == 2
        assert len(block_stride) == 2
        assert len(cell_size) == 2
        assert window_size[0] % cell_size[0] == 0
        assert window_size[1] % cell_size[1] == 0
        cell_number = (window_size[0] / cell_size[0], window_size[1] / cell_size[1])
        assert cell_number[0] >= block_size[0] >= 0
        assert cell_number[1] >= block_size[1] >= 0
        assert block_stride[0] > 0 and (cell_number[0] - block_size[0]) % block_stride[0] == 0
        assert block_stride[1] > 0 and (cell_number[1] - block_size[1]) % block_stride[1] == 0
        self.window_size = window_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbin = nbin
        self.full_angle = full_angle

    @property
    def ncell(self):
        return int(self.window_size[0] / self.cell_size[0]), int(self.window_size[1] / self.cell_size[1])

    @property
    def nblock(self):
        return int((self.ncell[0] - self.block_size[0]) / self.block_stride[0] + 1), int(
            (self.ncell[1] - self.block_size[1]) / self.block_stride[1] + 1)

    @property
    def bin_width(self):
        return self.full_angle / self.nbin

    @property
    def bin_value(self, index: int):
        assert 0 <= index < self.nbin
        return self.bin_width * index

    def compute_gradient(self, img):
        gx = cv2.Sobel(np.float32(img) / 255.0, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(np.float32(img) / 255.0, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        index = np.argmax(magnitude, axis=-1)
        magnitude = from_index(magnitude, index)
        orientation = from_index(orientation, index)
        return magnitude, orientation

    def compute_cell_histogram(self, img):
        hists = np.zeros((*self.ncell, self.nbin))
        magnitude, orientation = self.compute_gradient(img)
        grads = np.zeros((*self.window_size, 2))
        bins = np.zeros((*self.window_size, 2), dtype=np.int)
        for row in range(self.ncell[0]):
            for col in range(self.ncell[1]):
                angle = orientation[row * self.cell_size[0]:(row + 1) * self.cell_size[0],
                        col * self.cell_size[1]:(col + 1) * self.cell_size[1]] % self.full_angle
                weight = magnitude[row * self.cell_size[0]:(row + 1) * self.cell_size[0],
                         col * self.cell_size[1]:(col + 1) * self.cell_size[1]]
                for y in range(angle.shape[0]):
                    for x in range(angle.shape[1]):
                        if angle[y][x] < self.bin_width / 2 or angle[y][x] > self.full_angle - self.bin_width / 2:
                            left_bin = self.nbin - 1
                            right_bin = 0
                            if angle[y][x] < self.bin_width / 2:
                                left_weight = (self.bin_width / 2 + angle[y][x]) / self.bin_width
                                right_weight = (self.bin_width / 2 - angle[y][x]) / self.bin_width
                            else:
                                left_weight = (self.bin_width / 2 - (self.full_angle - angle[y][x])) / self.bin_width
                                right_weight = (self.bin_width / 2 + (self.full_angle - angle[y][x])) / self.bin_width
                        else:
                            pos = angle[y][x] - self.bin_width / 2
                            left_bin = int(pos / self.bin_width)
                            right_bin = left_bin + 1
                            left_weight = (pos - self.bin_width * left_bin) / self.bin_width
                            right_weight = (self.bin_width * right_bin - pos) / self.bin_width
                        left_weight *= weight[y][x]
                        right_weight *= weight[y][x]
                        hists[row][col][left_bin] += right_weight
                        hists[row][col][right_bin] += left_weight
                        grads[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][0] = left_weight
                        grads[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][1] = right_weight
                        bins[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][0] = left_bin
                        bins[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][1] = right_bin
        return hists, grads, bins

    def compute(self, img, block_norm_method = 'L2-Hys', visualize: bool = False):
        v = np.array([])
        hists, grads, bins = self.compute_cell_histogram(img)

        if visualize:
            assert self.full_angle == 180
            scale = 8
            window_size = (self.window_size[0] * scale, self.window_size[1] * scale)
            cell_size = (self.cell_size[0] * scale, self.cell_size[1] * scale)
            canvas = np.zeros((window_size[0], window_size[1], 3), dtype=np.float)
            radius = min(cell_size[0], cell_size[1]) // 2 - 2
            bins_arr = np.arange(self.nbin)
            angles = (np.pi * self.full_angle / 180) * (bins_arr + 0.5) / self.nbin
            stroke_length_x = radius * np.sin(angles)
            stroke_length_y = radius * np.cos(angles)
            for i in range(self.ncell[0]):
                for j in range(self.ncell[1]):
                    center_y = i * cell_size[0] + cell_size[0] // 2
                    center_x = j * cell_size[1] + cell_size[1] // 2
                    for k in range(self.nbin):
                        start_x = int(center_x + stroke_length_x[k])
                        start_y = int(center_y - stroke_length_y[k])
                        end_x = int(center_x - stroke_length_x[k])
                        end_y = int(center_y + stroke_length_y[k])
                        rr, cc = draw.line(start_y, start_x, end_y, end_x)
                        canvas[rr, cc] += hists[i, j, k]
            canvas_max = np.max(canvas)
            canvas_min = np.min(canvas)
            canvas = (canvas - canvas_min) / (canvas_max - canvas_min) * 255.0
            canvas = canvas.astype("uint8")

        for row in range(self.nblock[0]):
            for col in range(self.nblock[1]):
                temp = np.array([])
                for i in range(self.block_size[0]):
                    for j in range(self.block_size[1]):
                        temp = np.concatenate(
                            (temp, hists[i * self.block_stride[0] + row][j * self.block_stride[1] + col]))

                if block_norm_method[:2] == 'L2':
                    distance = np.sqrt(np.sum(temp ** 2) + 0.1 * len(temp))
                    temp /= distance
                    if block_norm_method == 'L2-Hys':
                        temp[temp > 0.2] = 0.2
                        distance = np.sqrt(np.sum(temp ** 2) + 0.001)
                        temp /= distance
                elif block_norm_method[:2] == 'L1':
                    distance = np.sum(temp)
                    temp /= distance
                    if block_norm_method == 'L1-sqrt':
                        temp = np.sqrt(temp)
                v = np.concatenate((v, temp))

        if visualize:
            return v, canvas
        else:
            return v

    @property
    def hog_dim(self):
        return self.nblock[0] * self.nblock[1] * self.block_size[0] * self.block_size[1] * self.nbin


def from_index(ndarray, index):
    if ndarray is not None and index is not None:
        flat_idx = np.arange(ndarray.size, step=ndarray.shape[-1]) + index.ravel()
        return ndarray.ravel()[flat_idx].reshape(*ndarray.shape[:-1])
    else:
        return None
