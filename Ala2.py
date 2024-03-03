import math
import cv2
import numpy as np


class Ala2:
    def __init__(self, R, G, B):
        self.H = None
        self.S = None
        self.I = None
        self.R = R
        self.G = G
        self.B = B

    def remove_red_eyes(self, point1, point2):
        self.rgb2hsi()
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        for i in range(min_y, min_y + height + 1):
            for j in range(min_x, min_x + width + 1):
                if (self.H[i, j] < math.pi * 1/ 4) or self.H[i, j] > -math.pi * 1 / 4:
                    if self.S[i, j] > 0.3:
                        self.S[i, j] = 0
        self.hsi2rgb()
        return self.R, self.G, self.B

    def rgb2hsi(self):
        m = self.R - (self.G + self.B) / 2
        n = cv2.multiply(self.R - self.G, self.R - self.G) + cv2.multiply(self.R - self.B, self.G - self.B)
        theta = np.zeros([self.R.shape[0], self.R.shape[1]])
        eps = 0.000001
        # 变换公式
        for i in range(self.R.shape[0]):
            for j in range(self.R.shape[1]):
                theta[i, j] = math.acos(m[i, j] / math.sqrt(n[i, j] + eps))
        self.H = theta.copy()
        mid = self.B > self.G
        self.H[mid] = 2 * math.pi * theta[mid]
        self.S = np.zeros([self.R.shape[0], self.R.shape[1]])
        for i in range(self.R.shape[0]):
            for j in range(self.R.shape[1]):
                self.S[i, j] = 1 - (3 * min(self.R[i, j], self.G[i, j], self.B[i, j]) / (
                        self.R[i, j] + self.G[i, j] + self.B[i, j] + eps))
        self.I = (self.R + self.G + self.B) / math.sqrt(3)

    def hsi2rgb(self):
        mid = math.pi * 2 / 3
        self.R = np.zeros([self.H.shape[0], self.H.shape[1]])
        self.G = np.zeros([self.H.shape[0], self.H.shape[1]])
        self.B = np.zeros([self.H.shape[0], self.H.shape[1]])
        for i in range(self.H.shape[0]):
            for j in range(self.H.shape[1]):
                if self.H[i, j] <= mid:
                    self.R[i, j] = (self.I[i, j] / math.sqrt (3)) * (
                            1 + self.S[i, j] * math.cos(self.H[i, j]) / math.cos(0.5 * mid - self.H[i, j]))
                    self.B[i, j] = (self.I[i, j]/ math.sqrt (3)) * (1 - self.S[i, j])
                    self.G[i, j] = math.sqrt (3) * self.I[i, j] - (self.R[i, j] + self.B[i, j])
                elif (self.H[i, j] <= mid * 2) and self.H[i, j] > mid:
                    self.H[i, j] = self.H[i, j] - mid
                    self.G[i, j] = (self.I[i, j] / math.sqrt (3)) * (
                            1 + self.S[i, j] * math.cos(self.H[i, j]) / math.cos(0.5 * mid - self.H[i, j]))
                    self.R[i, j] = (self.I[i, j]/ math.sqrt (3)) * (1 - self.S[i, j])
                    self.B[i, j] = 3 * self.I[i, j] - (self.R[i, j] + self.G[i, j])
                elif (self.H[i, j] <= mid * 2.5) and self.H[i, j] > mid * 2:
                    self.H[i, j] = self.H[i, j] - mid * 2
                    self.B[i, j] = (self.I[i, j] / math.sqrt (3)) * (
                            1 + self.S[i, j] * math.cos(self.H[i, j]) / math.cos(0.5 * mid - self.H[i, j]))
                    self.G[i, j] = self.I[i, j] * (1 - self.S[i, j])
                    self.R[i, j] = 3 * self.I[i, j] - (self.B[i, j] + self.G[i, j])

    def get_hsi(self):
        if self.H is not None and self.S is not None and self.I is not None:
            return self.H, self.S, self.I
        else:
            return None

    def get_rgb(self):
        if self.R is not None and self.G is not None and self.B is not None:
            return self.R, self.G, self.B
        else:
            return None
