# !/usr/bin/python3
import argparse
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sequence_similarity import compute_selfsimilarity, median_levelsets, recompute_at_fraction

import tifffile as tf

def voxel_map(image_list):
    # create a 3d array
    vos = np.stack(image_list) # stack along axis 0


def writeout(content, q_size, output_path, results, signal):
    q_size_text = str(q_size) if q_size >= 0 else 'all'
    ss_np_array = np.array(results[1])
    ssm_np_array = np.array(results[0])
    filename = Path(content).name
    ss_name = filename + '_' + q_size_text + '_' + 'entropy' + '.csv'
    ssm_name = filename + '_' + q_size_text + '_' + 'map' + '.csv'
    levelset_name = filename + '_' + q_size_text + '_' + 'levelset' + '.csv'
    file_entropy_name = filename + '_' + q_size_text + '_' + 'filelist' + '.csv'
    ss_name = os.path.join(output_path, ss_name)
    ssm_name = os.path.join(output_path, ssm_name)
    levelset_name = os.path.join(output_path, levelset_name)
    print(' ss %s ssm %s' % (ss_name, ssm_name))
    if Path(output_path).is_dir():
        np.savetxt(ss_name, ss_np_array, fmt='%-1.10f', delimiter=',')
        np.savetxt(ssm_name, ssm_np_array, fmt='%-1.10f', delimiter=',')
        if len(signal) == len(results[1]):
            np.savetxt(levelset_name, signal, fmt='%-1.10f', delimiter=',')

        if len(results) == 3: # was a dir of files, produce an array with index,filename,entropy
            print(f'{len(results[2])} files {len(results[1])} entropies ')
            if len(results[2]) == len(results[1]):
                file_entropy_name = os.path.join(output_path, file_entropy_name)
                header = 'Index,entropy,filename\n'
                csvF = open(file_entropy_name,'w')
                csvF.write(header)
                for ii in range(len(results[2])):
                    fn = Path(results[2][ii]).name
                    line = f'{ii},{results[1][ii]},{fn}\n'
                    csvF.write(line)
                csvF.close()



def main():
    parser = argparse.ArgumentParser(description='SelfSimilarator')
    parser.add_argument('--content', '-i', required=True,
                        help='Directory of sequentially numbered image files or TIF multipage file')
    parser.add_argument('--match', '-m', type=int, default=0, required=False, help='0 squared_ncv, 1 variation_of_information')
    parser.add_argument('--outpath', '-o', required=False, help='Path of output dir')
    parser.add_argument('--duration', '-d', type=int,required=True, help='Moving Temporal Window Size or -1 For All')
    parser.add_argument('--show', '-s', type=bool)
    parser.add_argument('--write', '-w', type=bool)
    parser.add_argument('--prefix', '-p', required=False,help='Image File Prefix, i.e. prefix0001.png')
    parser.add_argument('--type', '-t', required=True,help='Image File Extension')
    parser.add_argument('--levelset', '-l', type=int, default=0, required=False,help='Perform level setting with 1 / l fractions ')
    parser.add_argument('--voxels', '-v', type=int, default=0, required=False, help='Use voxels as sequence data ')
    parser.add_argument('--channel', '-c', type=int, default=-1, required=False, help='Channel to use if multichannel default converts to gray ')

    args = parser.parse_args()
    content_path = None
    output_path = None
    is_dir_of_files = True
    # if we have directory of images
    if Path(args.content).exists() and Path(args.content).is_dir():
        content_path = args.content
    # Single file containing a multi-page tif
    elif Path(args.content).exists() and Path(args.content).is_file():
        is_tif = Path(args.content).suffix == ('.tif')
        if is_tif:
            with tf.TiffFile(args.content) as tif:
                data = tif.asarray()
                if len(data.shape) > 2:
                    content_path = args.content
                    is_dir_of_files = False

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
    use_voxel = args.voxels

    if use_voxel and is_dir_of_files:
        print('Voxel Processing on input of directory of images is not supported at this time ')
        sys.exit(1)

    # compute_selfsimilarity(content_path, match_fn_index, q_size=-1, content_prefix='image', wild_dot_format='*.jpg',
    #                            use_voxels =False,  do_show=False, channel = -1):
    results = compute_selfsimilarity(content_path, args.match, q_size,
                                     args.prefix,content_glob, use_voxel, args.show, args.channel)

    if results is None:
        print('Input Error: Check ')
        sys.exit(1)



    # Optional Median Level Processing
    levelset = int(args.levelset)
    signal = []
    ss_np_array = np.array(results[1])
    ssm_np_array = np.array(results[0])
    if not (results is None) and levelset > 0 and levelset < 20:
        mlevel = median_levelsets(np.array(results[1]))
        print(mlevel['median'])
        signal = recompute_at_fraction(1.0 / levelset, mlevel['ranks'], ss_np_array, ssm_np_array)
        if args.show:
            plt.plot(signal)
            plt.show()

    if args.write:
        writeout(args.content,q_size, output_path, results, signal)


if __name__ == '__main__':
    main()



