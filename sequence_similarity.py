# !/usr/bin/python3
import argparse
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import numpy as np
import pwise
from skimage import io
import math

import tifffile as tf


'''
@todo:
1. Add progress report
2. Add parallel processing
3. Add input range
4. Add channel select
5. Add similarity function select
'''

# Normalized Correlation via cv2 or np.cov
# returns r^^2
# def correlation_coefficient__(image_a, image_b, options=None):
#     res = cv2.matchTemplate(img_as_float32(image_a), img_as_float32(image_b), cv2.TM_CCOEFF_NORMED)
#     c = res[0][0]
#     return c*c
#
#
# # returns r^^2
# def correlation_coefficient__(imagea, imageb, options=None):
#     image_a = np.ravel(imagea)
#     image_b = np.ravel(imageb)
#     c = np.cov(image_a, image_b)
#     d = np.sqrt(np.cov(image_a) * np.cov(image_b))
#     e = c / d
#     ee = e[0][1]
#     return ee * ee

## Computes normalized correlation^^2
def correlation_coefficient(image_a, image_b):
    product = np.mean((image_a - image_a.mean()) * (image_b - image_b.mean()))
    stds = image_a.std() * image_b.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return (product+1.0)/2

    # Check if wild_dot_format and multi_tif do not match
def format_is_tif(wild_dot_format):
    parts = wild_dot_format.split('.')
    if len(parts) != 2: return None
    is_tiff = (parts[1] == 'tif' or parts[1] == 'tiff' or parts[1] == 'TIF' or parts[1] == 'TIFF')
    return is_tiff

'''
 Compute ranks and median of entropy signal

 returns array of corresponding ranks ( integer ) and median of the entire signal

'''
def median_levelsets (sotime):
    if type(sotime) == list:
        entarray = np.asarray(sotime)
    else: entarray = sotime

    entmedian = np.median(entarray)
    entarray = np.subtract(entarray, entmedian)
    ranks = np.argsort(sotime)
    return {'ranks':ranks, 'median':entmedian}

def not_none(something):
    return not ( something is None)

def recompute_at_fraction(fraction, ranks, entropies, ss_map):
    assert(not_none(ranks) and not_none(entropies) and not_none(ss_map))
    assert(len(ranks) == len(entropies))
    signal_count = len(ranks)
    count =  math.floor(signal_count * fraction)
    signal = entropies.copy()
    for ii in range(signal_count):
        val = 0.0
        for index in range(count):
            jj = ranks[index]
            val += ss_map[jj][ii]
        val = val / count
        signal[ii] = val
    return signal



'''

Parameters:
  inputs: files ( sorted according to sequence if applicable ) or images
  q_size: duration in number of frames
  is_file: True if inputs are files paths and False if the input is ndarrays of images
  total_count: len of inputs
  do_show: plot results

returns:
  1D self-similarity calculation 
  2D self-similarity matrix
'''

def selfsimilarity(inputs, q_size, is_files, total_count, do_show):


    fItr = iter(range(total_count))

    buffer = []
    if q_size == -1: q_size=total_count
    else: q_size = q_size % total_count
    # create a pw unitary array
    # it sets the diagonal
    ssm = pwise.getPairWiseArray((q_size, q_size))

    def is_tiff(): return not is_files

    def getNextImage(iterator):
        item = next(iterator, None)
        if item is None: return None
        if is_tiff():
            return inputs[item]
        else:
            if not Path(inputs[item]).is_file():
                return None
            return io.imread (str(inputs[item]))
        assert(True) ## should not reach here

    def plotAndShow(ssm_, ss_, do_show):
        if not do_show: return
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        spacing = 0.005
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        rect_histx = [left, bottom_h, width, 0.2]

        # start with a rectangular Figure
        plt.figure(figsize=(8, 8))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.imshow(ssm_, cmap='gray')

        ax_histx = plt.axes(rect_histx)
        ax_histx.plot(ss_)
        plt.show()

    for pf in range(q_size):
        image = getNextImage(fItr)
        if image is None: break
        buffer.append(image)

    # create a deque for frame grabbing
    rb = deque(buffer)

    # fillup the ss with prefill comparisons
    for rr in range(q_size):
        for cc in range(rr + 1, q_size):
            r = correlation_coefficient(rb[rr], rb[cc])
            pwise.setPairWiseArrayPair(ssm, rr, cc, r)
    # if duration is all frame, we are done and it fall through.But computes the following median
    ss = 1.0 - pwise.getSelfSimilarity(ssm)
    # if duration was shorter than entire length, compute the median. Wasted if duration is all
    sotime = []
    sotime.append(np.median(ss))

    '''
    On New Frame: queue append frame, queue popleft   oldest old new, corr new against the rest and set, roll
    '''
    while (True):
        frame = getNextImage(fItr)
        if frame is None: break
        ## put in FIFO
        rb.append(frame)
        rb.popleft()
        for ff in range(q_size - 1):
            r = correlation_coefficient(frame, rb[ff])
            pwise.setPairWiseArrayPair(ssm, 0, ff + 1, r)
        np.roll(ssm, (q_size - 1) * q_size)
        ss = pwise.getSelfSimilarity(ssm)
        ssv = np.median(ss)
        sotime.append(1.0 - ssv)

    if(q_size == total_count):sotime = ss[0,:]
    plotAndShow(ssm, sotime, do_show)
    return (ssm,sotime)


def compute_selfsimilarity(content_path, q_size=-1, content_prefix='image', wild_dot_format='*.jpg', multi_tif=False,
                           do_show=False):
    _is_multi_tif = multi_tif and format_is_tif(wild_dot_format)
    source = content_path
    total_count = -1
    # handle directory of image files or a multi image tif file
    if _is_multi_tif == False:
        assert (Path(source).is_dir())
        p = Path(source).glob(wild_dot_format)
        files = [x for x in p if x.is_file()]
        files.sort(key=lambda f: int(f.name.strip(content_prefix).split('.')[0]))
        total_count = len(files)
        if total_count < 1: return None
        return selfsimilarity(files, q_size, True, total_count, do_show)
    else:  # is multi image tiff file
        assert (Path(source).is_file())
        mtif = tf.imread(source)
        assert (len(mtif.shape) > 2)
        total_count = mtif.shape[0]
        return selfsimilarity(mtif, q_size, False, total_count, do_show)























