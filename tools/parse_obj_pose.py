"""
this script provides a simple demo on how to interpret the object pose parameters saved in obj_fit.pkl file

Usage: python tools/parse_obj_pose.py -s [path to a sequence]

Author: Xianghui Xie
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os

import numpy as np

sys.path.append(os.getcwd())
import os.path as osp
import pickle as pkl
from scipy.spatial.transform import Rotation
from psbody.mesh import Mesh, MeshViewer
from data.frame_data import FrameDataReader

# path to the simplified mesh used for registration
simplified_mesh = {
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


def main(args):
    reader = FrameDataReader(args.seq_folder)
    category = reader.seq_info.get_obj_name(True)

    temp_simp, temp_full = Mesh(), Mesh()
    name = reader.seq_info.get_obj_name()
    # load simplified mesh template (the mesh used for registration)
    temp_simp.load_from_file(osp.join(args.seq_folder, f"../../objects/{simplified_mesh[name]}"))
    # load full template mesh
    temp_full.load_from_obj(osp.join(args.seq_folder, f"../../objects/{name}/{name}.obj"))
    # center the meshes
    center = np.mean(temp_simp.v, 0)
    temp_simp.v -= center
    temp_full.v -= center

    frames = np.random.choice(range(0, len(reader)), 5, replace=False)
    outfolder = osp.join(f'tmp/{reader.seq_name}')
    os.makedirs(outfolder, exist_ok=True)
    for idx in frames:
        idx = int(idx)
        pose_file = osp.join(reader.get_frame_folder(idx), category, f'{args.obj_name}/{category}_fit.pkl')
        data = pkl.load(open(pose_file, 'rb'))
        angle, trans = data['angle'], data['trans']
        rot = Rotation.from_rotvec(angle).as_matrix()

        # transform canonical mesh to fitting
        temp_simp_transformed = Mesh(temp_simp.v.copy(), temp_simp.f.copy())
        temp_simp_transformed.v = np.matmul(temp_simp_transformed.v, rot.T) + trans
        temp_full_transformed = Mesh(temp_full.v.copy(), temp_full.f.copy())
        temp_full_transformed.v = np.matmul(temp_full_transformed.v, rot.T) + trans

        obj_fit = reader.get_objfit(idx, args.obj_name)

        obj_fit.write_ply(osp.join(outfolder, f'{reader.frame_time(idx)}_fit.ply'))
        temp_full_transformed.write_ply(osp.join(outfolder, f'{reader.frame_time(idx)}_full_transformed.ply'))
        temp_simp_transformed.write_ply(osp.join(outfolder, f'{reader.frame_time(idx)}_simp_transformed.ply'))
    print(f'files saved to tmp/{reader.seq_name}')
    print('all done')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-on', '--obj_name', help='object fitting save name, for final dataset, use fit01',
                        default='fit01')

    args = parser.parse_args()

    main(args)