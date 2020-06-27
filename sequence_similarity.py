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



## Computes normalized correlation^^2
def correlation_coefficient(image_a, image_b):
    product = np.mean((image_a - image_a.mean()) * (image_b - image_b.mean()))
    stds = image_a.std() * image_b.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def compute_selfsimilarity(content_path, q_size=-1, content_prefix='image', content_format='*.jpg', do_show = False):

    src_dir = content_path
    assert (Path(src_dir).is_dir())
    p = Path(src_dir).glob(content_format)
    files = [x for x in p if x.is_file()]
    files.sort(key=lambda f: int(str(f.name).strip(content_prefix).split('.')[0]))
    fItr = iter(files)
    buffer = []
    if q_size == -1: q_size=len(files)
    else: q_size = q_size % len(files)
    # create a pw unitary array
    # it sets the diagonal
    ssm = pwise.getPairWiseArray((q_size, q_size))
    title = Path(src_dir).name

    def getNextFile(iterator):
        item = next(iterator, None)
        if item is None: return None
        if not Path(item).is_file():
            return None
        return str(item)

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
        ax_scatter.imshow(ssm_)

        ax_histx = plt.axes(rect_histx)
        ax_histx.plot(ss_)
        plt.show()

    for pf in range(q_size):
        file = getNextFile(fItr)
        if file is None: break
        buffer.append(io.imread(file))

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
    sotime.append(1.0 - np.median(ss))

    '''
    On New Frame: queue append frame, queue popleft   oldest old new, corr new against the rest and set, roll
    '''
    while (True):
        file = getNextFile(fItr)
        if file is None: break
        frame = io.imread(file)
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

    if(q_size == len(files)):sotime = ss[0,:]
    plotAndShow(ssm, sotime, do_show)
    return (ssm,sotime)

def main():
    parser = argparse.ArgumentParser(description='SelfSimilarator')
    parser.add_argument('--content', '-i', required=True,
                        help='Image Directory of sequentially numbered image files')
    parser.add_argument('--outpath', '-o', required=False, help='Path of output dir')
    parser.add_argument('--duration', '-d', type=int,required=True, help='Moving Temporal Window Size or -1 For All')
    parser.add_argument('--show', '-s', type=bool)
    parser.add_argument('--write', '-w', type=bool)
    parser.add_argument('--prefix', '-p', required=True,help='Image File Prefix, i.e. prefix0001.png')
    parser.add_argument('--type', '-t', required=True,help='Image File Extension')
    args = parser.parse_args()
    content_path = None
    output_path = None
    if Path(args.content).exists() and Path(args.content).is_dir():
        content_path = args.content
    if (content_path is None):
        str_format = " Error: Content Path %s is not valid "
        print(str_format % (content_path))
        sys.exit(1)
    if (not (args.outpath is None)) and Path(args.outpath).exists and Path(args.outpath).is_dir():
        output_path = args.outpath
    if output_path is None:
        output_path = os.path.dirname(os.path.realpath(__file__))

    content_glob = '*.' + args.type
    q_size = args.duration
    q_size_text = str(q_size) if q_size >= 0 else 'all'
    results = compute_selfsimilarity(content_path, q_size, args.prefix,content_glob, args.show)
    filename = Path(args.content).name
    ss_name = filename+ '_' + q_size_text + '_' + 'entropy' + '.csv'
    ssm_name = filename+ '_' +  q_size_text + '_' + 'map' + '.csv'
    ss_name = os.path.join(output_path, ss_name)
    ssm_name = os.path.join(output_path, ssm_name)
    print(' ss %s ssm %s' % (ss_name, ssm_name))
    ss_np_array = np.array(results[1])
    ssm_np_array = np.array(results[0])
    if args.write and Path(output_path).is_dir():
        np.savetxt(ss_name, ss_np_array, fmt='%-1.10f', delimiter=',')
        np.savetxt(ssm_name, ssm_np_array, fmt='%-1.10f', delimiter=',')




if __name__ == '__main__':
    main()



























