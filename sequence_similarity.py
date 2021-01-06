# !/usr/bin/python3
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import numpy as np
import pwise
from skimage import io
import math
import tifffile as tf
import logging
from skimage import color

from similarity_fns import information_variation, squared_ncv_cv2, squared_ncv_np, match_functions

logging.basicConfig(filename='sspy.log', format='%(asctime)s %(message)s', level=logging.INFO)

'''
@todo:
1. Add input range
'''



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


def selfsimilarity(inputs, is_dir_of_files , q_size, total_count, use_voxels, do_show, channel, mfunc):

    if type(inputs) is list:
        logging.info('is_dir_of_files: %r, use_voxel: %r, do_show: %r, channel: %d', is_dir_of_files, use_voxels,
                     do_show, channel)
    else:
        input_shape = inputs.shape
        z = input_shape[len(input_shape) -3] if len(input_shape) == 3 else -1
        x = input_shape[len(input_shape) -1]
        y = input_shape[len(input_shape) -2]
        logging.info('Input Shape: [%d,%d,%d] is_file %d q_size: %d, total count: %d ',
                 x, y, z, is_dir_of_files, q_size, total_count)

    is_image = True if (not is_dir_of_files) and (not use_voxels) else False
    fItr = iter(range(total_count))

    # create a buffer of samples.
    # number of images or number of 2d pixels for voxel processing
    buffer = []
    if q_size == -1: q_size=total_count
    else: q_size = q_size % total_count
    logging.info('Sample Count: %d, q_size: %d'%(total_count, q_size))

    # create a pw unitary array it sets the diagonal
    ssm = pwise.getPairWiseArray((q_size, q_size))

    pool = mp.Pool(mp.cpu_count())

    def getNextImage(iterator):
        item = next(iterator, None)
        if item is None: return None

        if is_dir_of_files:
            assert(Path(inputs[item]).is_file())
            logging.info('%s'%(str(inputs[item])))
            if channel == -1: # return gray of color
                return io.imread(str(inputs[item]), channel == -1)
            image = io.imread(str(inputs[item]), channel == -1)
            if image.ndim == 3:
                return image[:, :, channel]
            if image.ndim > 3:
                rgb = color.rgba2rgb(image)
                return rgb[:, :, channel]


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
    logging.info('Buffer Created Size: %d' % (len(buffer)))

    # generate random boolean mask the length of data
    # use p 0.75 for False and 0.25 for True
    # mask = np.random.choice([False, True], q_size, p=[0.90,0.10])

    # fillup the ss with prefill comparisons
    for rr in range(q_size):
        for cc in range(rr + 1, q_size):
            pr = [pool.apply(mfunc, args=( rb[rr], rb[cc]))]
            r = pr[0]
            pwise.setPairWiseArrayPair(ssm, rr, cc, r[0],r[1])

#        if (rr % (q_size//10)) == 0: logging.info(' Percent Done: %d'%(int(((rr*100.0)/q_size))))
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
        if frame is None:
            logging.info("Done")
            break
        ## put in FIFO
        rb.append(frame)
        rb.popleft()
        for ff in range(q_size - 1):
            pr = [pool.apply(mfunc, args=(frame, rb[ff]))]
            r = pr[0]
            pwise.setPairWiseArrayPair(ssm, 0, ff + 1, r[0],r[1])
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


def compute_selfsimilarity(content_path, match_fn_index, q_size=-1, content_prefix='image', wild_dot_format='*.tif',
                           use_voxels =False,  do_show=False, channel = -1):

    source = content_path
    total_count = -1
    mfunc = match_functions[match_fn_index]

    # handle directory of image files of wild_dot_format or a multi image tif file
    if Path(source).is_dir():
        p = Path(source).glob(wild_dot_format)
        files = [x for x in p if x.is_file()]
        files.sort(key=lambda f: int(f.name.strip(content_prefix).split('.')[0]))
        total_count = len(files)
        if total_count < 2: return None
        return selfsimilarity(files, True, q_size, total_count, use_voxels, do_show, channel, mfunc)
    elif Path(source).is_file():
        mtif = tf.imread(source)
        tif_shape = mtif.shape
        if len(tif_shape) < 3 or tif_shape[2] < 2:
            return None
        total_count = mtif.shape[0] if use_voxels == False else mtif.shape[1] * mtif.shape[2]
        return selfsimilarity(mtif, False, q_size, total_count, use_voxels, do_show, channel, mfunc)























