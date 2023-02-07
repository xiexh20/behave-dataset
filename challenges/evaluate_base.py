"""
base evaluator: load data and prepare SMPL and object meshes
Author: Xianghui Xie, Jan 20, 2023

Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
import os.path as osp
import pickle as pkl
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation


class BaseEvaluator:
    def __init__(self):
        self.m2mm = 1000. # meter to millimeter
        self.m2cm = 100. # meter to centimeter

        # for joint evaluation
        self.occ_thres = 0.3 # for object and joint recon
        self.sample_num = 6000 # surface sampling number for Chamfer distance computation

        # for object evaluation
        self.sample_obj = 10000
        self.re_max = 40. # rotation error max, for recall computation
        self.pe_max = 200. # pixel error max, for recall computation

    def eval(self, res_dir, gt_dir, outfile):
        """
        load GT and predictions, compute evaluation scores
        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return:
        """
        # step 1: check if the result data is complete
        raise NotImplemented

    def check_data(self, dict_gt:dict, dict_pred:dict):
        raise NotImplemented

    def load_data(self, gt_dir, res_dir):
        data_gt = pkl.load(open(osp.join(gt_dir, 'ref.pkl'), 'rb'))
        data_pred = pkl.load(open(osp.join(res_dir, 'results.pkl'), 'rb'))
        return data_gt, data_pred

    def write_errors(self, outfile, errs:dict):
        """
        save error dict to the output file
        :return:
        """
        str = ''
        for err in errs.keys():
            if not errs[err] == np.inf:
                str = str + err + ': {}\n'.format(errs[err])

        os.makedirs(osp.dirname(outfile), exist_ok=True)
        with open(outfile, 'w') as f:
            f.write(str)
        f.close()

        # print results
        s = 'Scores: '
        for k, v in errs.items():
            s += f'{k}={v:.4f}, '
        self.logging(s[:-2])
        self.logging(f"Evaluation done, scores saved to {outfile}")

    def get_obj_name(self, seq):
        obj_name = seq.split('_')[2]
        return obj_name

    def smpl2smplh_pose(self, poses):
        """add SMPLH hand pose to the SMPL pose"""
        assert poses.shape[-1] == 72, f'given pose shape {poses.shape} is not SMPL pose'
        p = np.zeros((poses.shape[0], 156))
        p[:, :69] = poses[:, :69]
        p[:, 111:114] = poses[:, 69:]
        poses = p
        return poses

    def logging(self, s):
        """
        logging function
        :param s: string to print
        :return:
        """
        utc_now = str(datetime.now())[:-3] # up to milliseconds
        utc_now = utc_now.replace(' ', "|")
        sys.stdout.write('{}: {}\n'.format(utc_now, s))
        sys.stdout.flush()

    def compute_overts_pred(self, data_pred, seq, temp_verts):
        """
        based on estimated object parameters, compute the object vertices
        :param data_pred: dict for all data
        :param seq: sequence name
        :param temp_verts: Nx3 object template vertices
        :return: LxNx3 a batch of object vertices for this seq
        """
        rot_pred = data_pred[seq]['obj_rots'].transpose(0, 2, 1)
        overts_pred = np.matmul(np.repeat(temp_verts[None], len(rot_pred), 0), rot_pred) \
                      + data_pred[seq]['obj_trans'][:, None]
        if 'obj_scales' in data_pred[seq]:
            overts_pred = overts_pred * np.array(data_pred[seq]['obj_scales'])[:, None, None]
        return overts_pred

    def compute_overts_gt(self, data_gt_params, seq, temp_verts):
        """
        compute GT object vertices
        :param data_gt_params: data dict for all GT parameters
        :param seq: sequence name
        :param temp_verts: Nx3 numpy arrray, object template vertices
        :return: LxNx3 a batch of object vertices
        """
        rot_gt = Rotation.from_rotvec(data_gt_params[seq]['obj_rots']).as_matrix()
        overts_gt = np.matmul(np.repeat(temp_verts[None], len(rot_gt), 0), rot_gt.transpose(0, 2, 1)) + \
                    data_gt_params[seq]['obj_trans'][:, None]
        return overts_gt