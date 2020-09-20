# !/usr/bin/python3
import argparse
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sequence_similarity import compute_selfsimilarity, median_levelsets, recompute_at_fraction

import tifffile as tf



def main():
    parser = argparse.ArgumentParser(description='SelfSimilarator')
    parser.add_argument('--content', '-i', required=True,
                        help='Directory of sequentially numbered image files or TIF multipage file')
    parser.add_argument('--outpath', '-o', required=False, help='Path of output dir')
    parser.add_argument('--duration', '-d', type=int,required=True, help='Moving Temporal Window Size or -1 For All')
    parser.add_argument('--show', '-s', type=bool)
    parser.add_argument('--write', '-w', type=bool)
    parser.add_argument('--prefix', '-p', required=False,help='Image File Prefix, i.e. prefix0001.png')
    parser.add_argument('--type', '-t', required=True,help='Image File Extension')
    parser.add_argument('--levelset', '-l', type=int, default=0, required=False,help='Perform level setting with 1 / l fractions ')

    args = parser.parse_args()
    content_path = None
    output_path = None
    multi_image_tif = False
    if Path(args.content).exists() and Path(args.content).is_dir():
        content_path = args.content
    elif Path(args.content).exists() and Path(args.content).is_file():
        is_tif = Path(args.content).suffix == ('.' + args.type)
        if is_tif:
            with tf.TiffFile(args.content) as tif:
                data = tif.asarray()
                if len(data.shape) > 2:
                    content_path = args.content
                    multi_image_tif = True

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
    results = compute_selfsimilarity(content_path, q_size, args.prefix,content_glob, multi_image_tif, args.show)
    ss_np_array = np.array(results[1])
    ssm_np_array = np.array(results[0])

    # Optional Median Level Processing
    levelset = int(args.levelset)
    signal = []
    if levelset > 0 and levelset < 20:
        mlevel = median_levelsets(ss_np_array)
        print(mlevel['median'])
        signal = recompute_at_fraction(1.0 / levelset, mlevel['ranks'], ss_np_array, ssm_np_array)
        if args.show:
            plt.plot(signal)
            plt.show()

    filename = Path(args.content).name
    ss_name = filename+ '_' + q_size_text + '_' + 'entropy' + '.csv'
    ssm_name = filename+ '_' +  q_size_text + '_' + 'map' + '.csv'
    levelset_name = filename+ '_' +  q_size_text + '_' + 'levelset' + '.csv'
    ss_name = os.path.join(output_path, ss_name)
    ssm_name = os.path.join(output_path, ssm_name)
    levelset_name = os.path.join(output_path, levelset_name)
    print(' ss %s ssm %s' % (ss_name, ssm_name))
    if args.write and Path(output_path).is_dir():
        np.savetxt(ss_name, ss_np_array, fmt='%-1.10f', delimiter=',')
        np.savetxt(ssm_name, ssm_np_array, fmt='%-1.10f', delimiter=',')
        if signal != []:
            np.savetxt(levelset, signal, fmt='%-1.10f', delimiter=',')




if __name__ == '__main__':
    main()



