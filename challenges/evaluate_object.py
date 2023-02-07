"""
object pose errors:
MSSD and MSPD: less dependent on surface sampling density
rotation error: for application where only the orientation is relevant but their is a global translation offset, e.g. joint human object recontruction
no translation error because it is already reflected in the MSSD

user can only submit results as pose parameters R, t, and scale (optional)

Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
import time

import trimesh
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_base import BaseEvaluator
import misc
import pose_error
import config
# from config import BEHAVE_CAM_K


class ObjectEvaluator(BaseEvaluator):
    def eval(self, res_dir, gt_dir, outfile):
        ""
        data_gt, data_pred = self.load_data(gt_dir, res_dir)

        data_complete = self.check_data(data_gt['annotations'], data_pred)
        if data_complete:
            err_dict = self.compute_errors(data_gt, data_pred)
        else:
            err_dict = {
                "MSSD": np.inf,
                "MSPD": np.inf,
                "RE": np.inf,
                "MSSD_AR": np.inf,
                "MSPD_AR": np.inf,
                "RE_AR": np.inf,
                "AR_all":np.inf
            }
        self.write_errors(outfile, err_dict)

    def check_data(self, dict_gt:dict, dict_pred:dict):
        """
        check if enough frames are predicted
        :param dict_gt:
        :param dict_pred:
        :return:
        """
        counts_gt, counts_pred = 0, 0
        for seq in dict_gt.keys():
            # check number of frames for evaluation based on occlusion ratio
            L1 = np.sum(np.array(dict_gt[seq]['occ_ratios'])>=self.occ_thres)
            counts_gt += L1
            frames_recon = dict_pred[seq]['frames']
            frames_gt = dict_gt[seq]['frames']
            valid = [1 if x in frames_gt else 0 for x in frames_recon]
            L2 = np.sum(valid)
            counts_pred += L2
        ratio = counts_pred / counts_gt
        self.logging(f"{ratio * 100:.2f}% frames are reconstructed")
        if ratio < 0.975:
            self.logging("Not enough frames are reconstructed for evaluation! Exiting...")
            return False
        else:
            self.logging(f"Start evaluating {counts_pred} examples.")
            return True

    def compute_errors(self, data_gt, data_pred):
        """

        :param data_gt:
        :param data_pred:
        :return: error dict
        """
        data_gt_params = data_gt['annotations']

        errors_mssd, errors_mspd, errors_re = [], [], []
        diameters = [] # object diameter for each example, used for computing recal
        start = time.time()
        cam_K = config.BEHAVE_CAM_K
        for seq in tqdm(sorted(data_gt_params.keys())):
            # self.logging(f'Evaluating sequence {seq}...')
            seq_start = time.time()
            obj_name = self.get_obj_name(seq)
            temp_faces = data_gt['templates'][obj_name]['faces']
            temp_verts = data_gt['templates'][obj_name]['verts']
            overts = trimesh.Trimesh(temp_verts, temp_faces, process=False).sample(self.sample_obj)
            diameter = misc.compute_diameter(overts)
            symms = data_gt['templates'][obj_name]['symmetries']

            # overts_gt = self.compute_overts_gt(data_gt_params, seq, overts)
            rot_gt = Rotation.from_rotvec(data_gt_params[seq]['obj_rots']).as_matrix()
            trans_gt = data_gt_params[seq]['obj_trans'][:, None] # Nx1x3
            rot_est = data_pred[seq]['obj_rots']
            overts_pred = self.compute_overts_pred(data_pred, seq, overts)

            occ_ratios = np.array(data_gt_params[seq]['occ_ratios'])

            # compute error for each example
            frames_pred = data_pred[seq]['frames']
            for idx, frame in enumerate(data_gt_params[seq]['frames']):
                if occ_ratios[idx] < self.occ_thres:
                    continue
                if frame not in frames_pred:
                    continue
                pidx = frames_pred.index(frame)  # find the recon index and get the data
                ov_est = overts_pred[pidx]

                errors_mssd.append(pose_error.mssd_verts(ov_est, rot_gt[idx], trans_gt[idx], overts, symms)*self.m2mm)
                errors_mspd.append(pose_error.mspd_verts(ov_est, rot_gt[idx], trans_gt[idx], cam_K, overts, symms))
                errors_re.append(pose_error.re_symm(rot_est[pidx], rot_gt[idx], symms))
                diameters.append(diameter*self.m2mm)

            end = time.time()
            self.logging(f'{seq} done, {end-seq_start:.4f} seconds.')
        self.logging(f'All sequences done, total {time.time()-start:.4f} seconds.')

        # compute recalls
        steps = np.arange(0.05, 0.51, 0.05)
        errors_dict = {
            "MSSD":errors_mssd,
            "MSPD":errors_mspd,
            "RE":errors_re
        }
        err_max = {
            "MSSD":np.array(diameters)*1.0,
            "MSPD":self.pe_max,
            "RE":self.re_max
        }
        recalls = {k:[] for k in errors_dict.keys()}
        for step in steps:
            for err in errors_dict.keys():
                thres = err_max[err] * step
                mask = errors_dict[err] < thres
                recall = np.sum(mask) / len(mask)
                recalls[err].append(recall)
        # average recall
        recall_dict = {f'{k}_AR':np.mean(v) for k, v in recalls.items()}
        avgerr_dict = {k:np.mean(v) for k, v in errors_dict.items()}
        err_dict = {
            **avgerr_dict,
            **recall_dict,
            "AR_all":np.mean(np.array(list(recall_dict.values()), dtype=float)) # for final ranking
        }

        return err_dict


if __name__ == '__main__':
    # Default execution for codalab
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Process reference and results directory
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, 'scores.txt')

    evaluator = ObjectEvaluator()
    evaluator.eval(submit_dir, truth_dir, output_filename)



