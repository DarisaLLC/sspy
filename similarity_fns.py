# !/usr/bin/python3
import numpy as np
import cv2
from skimage.util import img_as_float32
from skimage.metrics import (adapted_rand_error,variation_of_information)
from skimage.metrics import normalized_root_mse


def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

# Similarity Functions: All return a list of two scores:

# Normalized Correlation via cv2 or np.cov
# returns r^^2
def information_variation(image_a, image_b, options=None):
    vv = variation_of_information(image_a, image_b)
    v0 = clamp(vv[0], 0.0, 1.0)
    v1 = clamp(vv[1], 0.0, 1.0)
    return [v0, v1]

def root_nmse(image_a, image_b, options=None):
    v = normalized_root_mse(image_a, image_b)
    v = clamp(v, 0.0, 1.0)
    return [v,v]


# returns r^^2
def squared_ncv_cv2(image_a, image_b, options=None):
    res = cv2.matchTemplate(img_as_float32(image_a), img_as_float32(image_b), cv2.TM_CCOEFF_NORMED)
    c = res[0][0]
    c = c * c
    return [c, c]

def squared_nsd_cv2(image_a, image_b, options=None):
    res = cv2.matchTemplate(img_as_float32(image_a), img_as_float32(image_b), cv2.TM_SQDIFF_NORMED)
    c = res[0][0]
    c = c * c
    return [c, c]

def ncv_np(imagea, imageb, options=None):
    image_a = np.ravel(imagea)
    image_b = np.ravel(imageb)
    c = np.cov(image_a, image_b)
    d = np.sqrt(np.cov(image_a) * np.cov(image_b))
    e = c / d
    ee = e[0][1]
    return ee * ee



# Computes normalized correlation^^2
def squared_ncv_np(image_a, image_b, options=None):

    product = np.mean((image_a - image_a.mean()) * (image_b - image_b.mean()))
    stds = image_a.std() * image_b.std()
    if stds == 0:
        return [0,0]
    else:
        product /= stds
        product = product * product  # error2 #(product+1.0)/2
        if product > 1.0: product = 1.0
        return [product,product]

def unit_ncv (image_a, image_b):
    product = np.mean((image_a - image_a.mean()) * (image_b - image_b.mean()))
    stds = image_a.std() * image_b.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return (product+1.0)/2


match_functions = {0:squared_ncv_cv2, 1:information_variation, 2:root_nmse, 3: unit_ncv, 4:squared_nsd_cv2}


