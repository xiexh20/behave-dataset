"""
load color and depth videos and save them to images
RGB is saved as jpg file by default
"""
import sys, os
sys.path.append(os.getcwd())
from tqdm import tqdm
from glob import glob
import numpy as np
from os.path import join, basename, dirname
from data.seq_utils import save_seq_info
from data.const import _sub_gender, OBJ_NAMES
from data.utils import availabe_kindata, save_color_depth
from data.video_reader import ColorDepthController, VideoController


def save_seq_info_data(seq):
    seq_name = basename(seq)
    ss = seq_name.split('_')
    obj_name = seq_name.split('_')[2]
    date, subj = ss[0], ss[1]
    assert obj_name in OBJ_NAMES, f'invalid object name {obj_name} found!'

    # modify here if the path to calibration files are different
    config = f'../../calibs/{date}/config'
    intrinsic = f'../../calibs/intrinsics'
    empty = f'../../calibs/{date}/background'
    beta = None
    gender = _sub_gender[subj]
    kids = [x for x in range(4)]
    save_seq_info(seq, config, intrinsic, obj_name, gender, empty, beta, kids)


def main(args):
    input_color = args.video
    kids, comb = availabe_kindata(input_color, kinect_count=4)
    print("Available kinects for sequence {}: {}".format(basename(input_color), kids))
    kinect_count = len(kids)

    # load videos
    video_prefix = basename(input_color).split('.')[0]
    video_folder = dirname(input_color)
    if args.nodepth:
        controllers = [VideoController(os.path.join(video_folder, f'{video_prefix}.{k}.color.mp4')) for k in kids]
    else:
        controllers = [ColorDepthController(os.path.join(video_folder, video_prefix), k) for k in kids]
    end_time = np.min([controllers[x].end_time() for x in range(kinect_count)]) if args.tend is None else args.tend
    start_time = args.tstart
    fps = args.fps
    times = np.arange(start_time, end_time - 1./fps, 1./fps).tolist()

    out_dir = join(args.outpath, video_prefix)
    os.makedirs(out_dir, exist_ok=True)
    ext = args.ext

    loop = tqdm(times)
    loop.set_description(f"processing {video_prefix}")
    for t in loop:
        frame_folder = join(out_dir, 't{:08.3f}'.format(t))
        if not os.path.exists(frame_folder):
            os.mkdir(frame_folder)
        files = glob(join(frame_folder, f'*.color.{ext}'))
        depth_files = glob(join(frame_folder, f'*.depth.png'))
        if len(files) == kinect_count and not args.redo and len(depth_files) == kinect_count:
            print("frame t{:08.3f} already exist, skipped".format(t))
            continue

        # first choose the closest actual time, then use that timestamp to select multi-view images
        actual_times = np.array([controllers[x].get_closest_time(t) for x in kids])
        best_kid = np.argmin(np.abs(actual_times - t))
        actual_time = actual_times[best_kid]

        for k in kids:
            if args.nodepth:
                color, depth = controllers[k].get_closest_frame(actual_time), None
            else:
                color, depth = controllers[k].get_closest_frame(actual_time)
            save_color_depth(frame_folder, color, depth, k, ext=ext, color_only=args.nodepth)

    for ctrl in controllers:
        ctrl.close()
    # save sequence meta information
    save_seq_info_data(out_dir)
    print(f"seq images saved to {out_dir}, all done")



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('video', help='path to a video file')
    parser.add_argument('outpath', help='root path to all sequences')
    parser.add_argument('-ext', default='jpg', help='file extension for the RGB image', choices=['jpg', 'png'])
    parser.add_argument('-fps', type=int, default=30, help='generate frames at which fps')
    parser.add_argument('-tstart', type=float, default=3.0, help='first frame time')
    parser.add_argument('-tend', type=float, default=None, help='last frame time')
    parser.add_argument('-delay', default=False, action='store_true')
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-nodepth', default=False, action='store_true',
                        help='save depth images or not, if not, will not load depth video')

    args = parser.parse_args()

    main(args)

