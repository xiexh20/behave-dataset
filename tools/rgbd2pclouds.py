"""
demo code to convert RGBD images to point clouds, using the segmentation mask
example usage:
python tools/rgbd2pclouds.py BEHAVE_SEQ_ROOT/Date05_Sub05_chairwood -t obj
"""
import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join, dirname, isfile
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform
from tools.pc_filter import PCloudsFilter
from data.utils import write_pointcloud


def main(args):
    reader = FrameDataReader(args.seq, check_image=False)
    kin_transform = KinectTransform(args.seq)
    rotations, translations = kin_transform.local2world_R, kin_transform.local2world_t
    start = args.start
    end = reader.cvt_end(args.end)
    kids = reader.seq_info.kids

    # specify output dir
    outroot = dirname(args.seq) if args.out is None else args.out
    out_seq = join(outroot, reader.seq_name)
    target = 'person' if args.target == 'person' else reader.seq_info.get_obj_name(convert=True)

    # point cloud processing
    filter = PCloudsFilter()

    for i in tqdm(range(start, end)):
        rgb_imgs = reader.get_color_images(i, kids)
        dmaps = reader.get_depth_images(i, kids)

        out_frame = join(out_seq, reader.frame_time(i), target)
        os.makedirs(out_frame, exist_ok=True)
        outfile = join(out_frame, f'{target}.ply')
        if isfile(outfile) and not args.redo:
            continue
        masks = []
        complete = True
        for kid in kids:
            mask = reader.get_mask(i, kid, args.target)
            if mask is None:
                mask = np.zeros((1536, 2048)).astype(bool)
                complete = False
            masks.append(mask)
        if not complete:
            print(f"Warning: the {target} mask for frame {reader.frame_time(i)} is not complete!")
            continue
        pc_all, color_all = [], []
        for kid, mask in enumerate(masks):
            if np.sum(mask) == 0:
                continue
            depth_masked = np.copy(dmaps[kid])  # the rgb mask can be directly applied to depth image
            depth_masked[~mask] = 0

            pc, pc_color =  kin_transform.intrinsics[kid].dmap2colorpc(rgb_imgs[kid], depth_masked)
            if len(pc) == 0:
                continue
            # local camera to world transform
            pc_world = np.matmul(pc, rotations[kid].T) + translations[kid]
            pc_all.append(pc_world)
            color_all.append(pc_color)
        if len(pc_all) == 0:
            print("no mask in", reader.get_frame_folder(i))
            continue
        # do filtering
        pc_all = np.concatenate(pc_all, 0)
        color_all = np.concatenate(color_all, 0)
        pc_filtered, color_filtered = filter.filter_pclouds(pc_all, color_all, target)

        if pc_filtered is None:
            print("{} filtered out completed".format(outfile))
            continue
        write_pointcloud(outfile, pc_filtered, color_filtered)

    print('all done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('seq')
    parser.add_argument('-config', default=None, help='if not given, load from info.json file')
    parser.add_argument('-o', '--out', default=None, help='if not given, save to the original sequence path')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=5)
    parser.add_argument('-t', '--target', choices=['person', 'obj'])
    parser.add_argument('-redo', default=False, action='store_true')

    args = parser.parse_args()

    main(args)