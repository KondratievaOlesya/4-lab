import numpy as np
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
from math import exp, pi, sqrt, cos, sin
from sklearn import preprocessing

COLS = 1
ROWS = 0


def scScale(img, m, n):
    M, N = img.shape
    # l, m, n = getL(M, N)
    # m = int(M/l)
    # n = int(N/l)
    result = np.zeros((m + 1, n + 1))
    l = max(int(M / m), int(N / n))
    for i in range(0, M, l):
        for j in range(0, N, l):
            sum = 0
            count = 0
            for k in range(i, min(i + l, M)):
                for h in range(j, min(j + l, N)):
                    sum += img[k, h]
                    count += 1
            result[int(i / l), int(j / l)] = sum / count
    return result


def DFT(img, p):
    M, N = img.shape
    FpmCos = np.zeros((p, M))
    FpmSin = np.zeros((p, M))
    FnpCos = np.zeros((N, p))
    FnpSin = np.zeros((N, p))
    for i in range(0, p):
        for j in range(0, M):
            FpmCos[i][j], FpmSin[i][j] = cos(2 * pi / M * i * j), sin(2 * pi / M * i * j)

    for i in range(0, N):
        for j in range(0, p):
            FnpCos[i][j], FnpSin[i][j] = cos(2 * pi / N * i * j), sin(2 * pi / N * i * j)
    FXcos = (FpmCos.dot(img))
    FXsin = (FpmSin.dot(img))
    Creal = FXcos.dot(FnpCos) - FXsin.dot(FnpSin)
    Cimag = FXcos.dot(FnpSin) + FXsin.dot(FnpCos)
    tmp = np.square(Creal) + np.square(Cimag)
    C = np.sqrt(tmp)
    return C


def DCT(img, p):
    M, N = img.shape
    Tpm = np.zeros((p, M))
    Tnp = np.zeros((N, p))

    for j in range(0, M):
        Tpm[0, j] = 1 / sqrt(M)
    for i in range(1, p):
        for j in range(0, M):
            Tpm[i, j] = sqrt(2 / M) * cos((pi * (2 * j + 1) * i) / (2 * M))

    for i in range(0, N):
        Tnp[i, 0] = 1 / sqrt(N)
    for i in range(0, N):
        for j in range(0, p):
            Tnp[i, j] = sqrt(2 / N) * cos((pi * (2 * i + 1) * j) / (2 * N))

    C = (Tpm.dot(img)).dot(Tnp)
    return C


def histogram(img, BIN):
    Hi = [0 for _ in range(256)]
    M, N = img.shape
    for i in range(0, M):
        for j in range(0, N):
            Hi[img[i, j]] += 1

    Hb = [0 for _ in range(BIN)]
    for i in range(0, BIN):
        for j in range(int(i * 256 / BIN), int((i + 1) * 256 / BIN)):
            Hb[i] += Hi[j]
    HbNorm = [Hb[i] / (M * N) for i in range(BIN)]
    return [Hi, Hb, HbNorm]


def gradient(img, W, S, type=COLS):
    M, N = img.shape
    result = []

    if type == COLS:
        lastRow = img[0:W]
        for i in range(S, M - W + 1, S):
            row = img[i:(i + W)]
            diff = abs(np.linalg.norm(lastRow - row))
            lastRow = row
            result.append(diff)
        return result
    elif type == ROWS:
        lastCol = img[:, 0:W]
        for i in range(S, N - W, S):
            col = img[:, i:i + W]
            diff = abs(np.linalg.norm(lastCol - col))
            lastCol = col
            result.append(diff)
        return result


def distance(test, template):
    return abs(np.linalg.norm(np.array(test) - np.array(template)))


m, n, pDFT, pDCT, BIN, S, W = [28, 24, 10, 100, 16, 3, 12]
img = cv2.imread('orl_faces/s5/4.pgm')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray1 = np.float32(gray)
#dft = DCT(gray, pDCT)
# cv2.imshow('image', img)
# cv2.imshow('dft', np.uint16(dft))
# cv2.waitKey(0)

#Hi, Hb, HbNorm = histogram(gray, BIN)


grad = gradient(gray, W, S)
count = [i for i in range(1, len(grad) + 1)]
fig = plt.subplot()
mean = sum(grad)/len(grad)

fig.plot(count, np.array(grad)/mean)
plt.show()
imgToShow = Image.fromarray(gray)
imgToShow.show()
