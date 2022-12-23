"""
loads calibrations
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys
sys.path.append("/")
import json
from os.path import join, basename, dirname, isfile
import numpy as np
import cv2
from PIL import Image
from data.kinect_calib import KinectCalib


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_intrinsics(intrinsic_folder, kids):
    """
    kids: list of kinect id that should be loaded
    """
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]

    return kinects


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]['rotation']).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]['translation']) for x in kids]
    return rotations, translations


def load_kinects(intrinsic_folder, config_folder, kids):
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    pose_files = [join(config_folder, f"{x}/config.json") for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]
    return kinects


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


def availabe_kindata(input_video, kinect_count=3):
    # all available kinect videos in this folder, return the list of kinect id, and str representation
    fname_split = os.path.basename(input_video).split('.')
    idx = int(fname_split[1])
    kids = []
    comb = ''
    for k in range(kinect_count):
        file = input_video.replace(f'.{idx}.', f'.{k}.')
        if os.path.exists(file):
            kids.append(k)
            comb = comb + str(k)
        else:
            print("Warning: {} does not exist in this folder!".format(file))
    return kids, comb


def save_color_depth(out_dir, color, depth, kid, color_only=False, ext='jpg'):
    color_file = join(out_dir, f'k{kid}.color.{ext}')
    # cv2.imwrite(color_file, color[:, :, ::-1])
    Image.fromarray(color).save(color_file)
    if not color_only:
        depth_file = join(out_dir, f'k{kid}.depth.png')
        cv2.imwrite(depth_file, depth)

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

def get_template_path(behave_path, obj_name):
    path = join(behave_path, "objects", _mesh_template[obj_name])
    if not isfile(path):
        print(path, 'does not exist, please check input parameters!')
        raise ValueError()
    return path


def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    from psbody.mesh import Mesh
    scan = Mesh()
    # print(scan_path)
    scan.load_from_file(scan_path)
    if cent:
        center = np.mean(scan.v, axis=0)
        verts_centerd = scan.v - center
        scan.v = verts_centerd

    return scan


def load_template(obj_name, cent=True, dataset_path=None):
    assert dataset_path is not None, 'please specify BEHAVE dataset path!'
    temp_path = get_template_path(dataset_path, obj_name)
    return load_scan_centered(temp_path, cent)

def write_pointcloud(filename,xyz_points,rgb_points=None):
    """
    updated on March 22, use trimesh for writing
    """
    import trimesh
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    outfolder = dirname(filename)
    os.makedirs(outfolder, exist_ok=True)
    pc = trimesh.points.PointCloud(xyz_points, rgb_points)
    pc.export(filename)