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
from skimage.feature import register_translation
import tifffile as tf
import logging

logging.basicConfig(filename='sspy.log', format='%(asctime)s %(message)s', level=logging.INFO)

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
    # shift, error, diffphase = register_translation(image_a, image_b, space='real')
    # diff = (image_a - image_a.mean()) - (image_b - image_b.mean())
    # maxd = np.max(diff)
    # error2 = np.sqrt(np.mean(diff ** 2)) / maxd


    product = np.mean((image_a - image_a.mean()) * (image_b - image_b.mean()))
    stds = image_a.std() * image_b.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        product = product * product #error2 #(product+1.0)/2
        if product > 1.0: product = 1.0
        return product

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
  inputs: files ( sorted according to sequence if applicable ) or images or voxels
  content_type image_files, image_stack, voxel_stack 
  q_size: duration in number of frames
   
  do_show: plot results

returns:
  1D self-similarity calculation 
  2D self-similarity matrix
'''


def selfsimilarity(inputs, is_file , q_size, total_count, use_voxels, do_show):
    input_shape = inputs.shape
    z = input_shape[2] if len(input_shape) == 3 else -1
    x = input_shape[len(input_shape) -1]
    y = input_shape[len(input_shape) -2]

    logging.info('Input Shape: [%d,%d,%d] is_file %d q_size: %d, total count: %d ',
                 x, y, z, is_file, q_size, total_count)


    if is_file and use_voxels: return None
    is_image = True if (not is_file) and (not use_voxels) else False


    fItr = iter(range(total_count))

    buffer = []
    if q_size == -1: q_size=total_count
    else: q_size = q_size % total_count
    # create a pw unitary array
    # it sets the diagonal
    ssm = pwise.getPairWiseArray((q_size, q_size))



    def getNextImage(iterator):
        item = next(iterator, None)
        if item is None: return None

        if is_file:
            assert(Path(inputs[item]).is_file())
            return io.imread (str(inputs[item]))
        elif is_image:
            return inputs[item]
        elif use_voxels:
            uri = np.unravel_index(item, inputs.shape)
            return inputs[:,uri[1],uri[2]]

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
    if use_voxels:
        ishape = inputs.shape
        print(ishape)
        ssm = sotime.reshape(ishape[1], ishape[2])
    plotAndShow(ssm, sotime, do_show)
    return (ssm,sotime)


def compute_selfsimilarity(content_path, q_size=-1, content_prefix='image', wild_dot_format='*.jpg',
                           use_voxels =False,  do_show=False):

    source = content_path
    total_count = -1

    # handle directory of image files of wild_dot_format or a multi image tif file
    if Path(source).is_dir():
        p = Path(source).glob(wild_dot_format)
        files = [x for x in p if x.is_file()]
        files.sort(key=lambda f: int(f.name.strip(content_prefix).split('.')[0]))
        total_count = len(files)
        if total_count < 2: return None
        return selfsimilarity(files, True, q_size, total_count, use_voxels, do_show)
    elif Path(source).is_file():
        mtif = tf.imread(source)
        tif_shape = mtif.shape
        if len(tif_shape) < 3 or tif_shape[2] < 2:
            return None
        total_count = mtif.shape[0] if use_voxels == False else mtif.shape[1] * mtif.shape[2]
        return selfsimilarity(mtif, False, q_size, total_count, use_voxels, do_show)























