import numpy as np
import os
import math
from math import log10
from scipy.fftpack import dct
from skimage.color import rgb2ycbcr


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def vari(AA):
    d = np.var(AA[:])*AA.size
    return d

def maskeff(z,zdct):
#Calculation of Enorm value(see[1])
    m = 0
    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
                [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
                [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
                [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
                [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
                [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
                [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
                [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    # see an explanation in [1]
    for k in range(1, 8):
        for l in range(1, 8):
            if (k != 1) | (l != 1):
                m = m + (zdct[k-1, l-1]**2)*MaskCof[k-1, l-1]
    pop = vari(z)
    if pop != 0:
        pop = (vari(z[0:4, 0:4]) + vari(z[0:4, 4:8]) + vari(z[4:8, 4:8]) + vari(z[4:8, 0:4]))/pop
    m = math.sqrt(m*pop)/32

    return m

def unpsn(aa):
    d = (255 * 255) / (10 ** (aa / 10))
    return d

def psnrhma(img1, img2):
    step = 8
    LenXY = img1.shape
    LenY = LenXY[0]
    LenX = LenXY[1]
    CSFCof = np.array([[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
                       [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
                       [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
                       [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
                       [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
                       [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
                       [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
                       [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]])

    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
                        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
                        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
                        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
                        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
                        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
                        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
                        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    #delt = ((np.sum(img1).astype(np.float64)) - (np.sum(img2).astype(np.float64))) / img1.size
    delt = ((np.sum(img1)) - (np.sum(img2))) / img1.size
    #img2m = img2.astype(np.float64) + delt
    img2m = img2 + delt
    mean1 = np.mean(img1)
    mean2 = np.mean(img2m)
    tmp = (img1-mean1) * (img2m-mean2)
    sq = np.var(img2m)
    l = np.sum(tmp) / tmp.size / sq
    if l<1:
        KofContr = 0.002
    else:
        KofContr = 0.25
    img3m = mean2 + (img2m-mean2)*l
    S3 = (np.sum((img2m-img1)**2) - np.sum((img3m-img1)**2)) / img1.size

    S1 = 0
    S2 = 0
    Num = 0
    SS1 = 0
    SS2 = 0
    X = 1
    Y = 1

    while Y <= LenY - 7:
        while X <= LenX - 7:
            A = img1[Y - 1:Y - 1 + 8, X - 1:X - 1 + 8]
            B = img2m[Y - 1:Y - 1 + 8, X - 1:X - 1 + 8]
            B2 = img3m[Y - 1:Y - 1 + 8, X - 1:X - 1 + 8]
            A_dct = dct2(A)
            B_dct = dct2(B)
            B_dct2 = dct2(B2)
            MaskA = maskeff(A, A_dct)
            MaskB = maskeff(B, B_dct)
            if MaskB > MaskA:
                MaskA = MaskB

            X = X + step
            for k in range(1, 8):
                for l in range(1, 8):
                    u = abs(A_dct[k - 1, l - 1] - B_dct[k - 1, l - 1])
                    u2 = abs(A_dct[k - 1, l - 1] - B_dct2[k - 1, l - 1])
                    S2 = S2 + (u * CSFCof[k - 1, l - 1]) ** 2
                    SS2 = SS2 + (u2 * CSFCof[k - 1, l - 1]) ** 2
                    if (k != 1) or (l != 1):
                        if u < MaskA / MaskCof[k - 1, l - 1]:
                            u = 0
                        else:
                            u = u - MaskA / MaskCof[k - 1, l - 1]
                        if u2 < MaskA / MaskCof[k - 1, l - 1]:
                            u2 = 0
                        else:
                            u2 = u2 - MaskA / MaskCof[k - 1, l - 1]
                    S1 = S1 + (u * CSFCof[k - 1, l - 1]) ** 2
                    SS1 = SS1 + (u2 * CSFCof[k - 1, l - 1]) ** 2
                    Num = Num + 1
        X = 1
        Y = Y + step

    if Num != 0:
        S1 = S1 / Num
        S2 = S2 / Num
        SS1 = SS1 / Num
        SS2 = SS2 / Num
        delt = delt ** 2

        if S1 > SS1:
            S1 = SS1 + (S1 - SS1) * KofContr
        S1 = S1 + 0.04 * delt
        if S2 > SS2:
            S2 = SS2 + (S2 - SS2) * KofContr
        S2 = S2 + 0.04 * delt

        if S1 == 0:
            phvsm = 100000  # img1 and img2 are visually undistingwished
        else:
            phvsm = 10 * log10(255 * 255 / S1)

        if S2 == 0:
            phvs = 100000  # img1 and img2 are identical
        else:
            phvs = 10 * log10(255 * 255 / S2)

    return phvsm, phvs

def color_psnrhma(img1,img2):

    a = rgb2ycbcr(img1).astype(int)

    a1 = a[:, :, 0].astype(np.float64)
    a2 = a[:, :, 1].astype(np.float64)
    a3 = a[:, :, 2].astype(np.float64)

    b = rgb2ycbcr(img2).astype(int)
    b1 = b[:, :, 0].astype(np.float64)
    b2 = b[:, :, 1].astype(np.float64)
    b3 = b[:, :, 2].astype(np.float64)

    p11, p12 = psnrhma(a1, b1)
    p21, p22 = psnrhma(a2, b2)
    p31, p32 = psnrhma(a3, b3)

    p11 = unpsn(p11)
    p12 = unpsn(p12)
    p21 = unpsn(p21)
    p22 = unpsn(p22)
    p31 = unpsn(p31)
    p32 = unpsn(p32)

    S1 = (p11 + (p21 + p31) * 0.5) / 2
    S2 = (p12 + (p22 + p32) * 0.5) / 2

    if S1 == 0:
        p_hvs_m = 100000  # img1 and img2 are visually undistingwished
    else:
        p_hvs_m = 10*log10(255*255/S1)

    if S2 == 0:
        p_hvs = 100000  # img1 and img2 are identical
    else:
        p_hvs = 10*log10(255*255/S2)

    return p_hvs_m, p_hvs
