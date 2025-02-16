"""
evaluate object errors
"""
import sys, os
import time
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'program')) # path for remote server
sys.path.append(os.path.join(os.getcwd(), 'challenges'))
import trimesh
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from lib.evaluate_base import BaseEvaluator
from lib import pose_error, misc, config


class ObjectEvaluator(BaseEvaluator):
    def eval(self, res_dir, gt_dir, outfile):
        ""
        data_gt, data_pred = self.load_data(gt_dir, res_dir)

        data_complete = self.check_data(data_gt, data_pred)
        if data_complete:
            err_dict = self.compute_errors(data_gt, data_pred)
        else:
            err_dict = {}
            for dname in config.DATASET_NAMES:
                ed_i = {f'{k}_{dname}':np.inf for k in ['MSSD', 'MSPD', 'RE', 'AR_all']}
                err_dict.update(**ed_i)

        self.write_errors(outfile, err_dict)

    def check_data(self, dict_gt:dict, dict_pred:dict):
        """
        check if enough frames are predicted
        :param dict_gt:
        :param dict_pred:
        :return:
        """
        if len(dict_pred.keys()) < 0.98 * len(dict_gt['annotations'].keys()):
            self.logging("Not enough frames are reconstructed for evaluation! Exiting...")
            return False
        else:
            self.logging(f"Start evaluating {len(dict_pred.keys())} examples.")
            return True

    def compute_errors(self, data_gt, data_pred):
        """
        simply iterate over all frames and compute per-dataset average errors

        :param data_gt:
        :param data_pred:
        :return: error dict, key-value pair to be show in leaderboard
        """
        start = time.time()

        error_dict = {}
        for k in tqdm(sorted(data_pred.keys())):
            if k not in data_gt['annotations']:
                self.logging(f'image id {k} not found in GT data!')
                continue
            dname, gender, obj_name = data_gt['annotations'][k]['meta'] # obj id indicates the object template
            # temp_faces = data_gt['templates'][obj_name]['faces']
            # temp_verts = data_gt['templates'][obj_name]['verts']
            overts = data_gt['templates'][obj_name]['verts']
            # overts = trimesh.Trimesh(temp_verts, temp_faces, process=False).sample(self.sample_obj)
            diameter = misc.compute_diameter(overts)
            symms = data_gt['templates'][obj_name]['symmetries']

            # overts_gt = self.compute_overts_gt(data_gt_params, seq, overts)
            rot_gt = data_gt["annotations"][k]['obj_rot']
            trans_gt = data_gt["annotations"][k]['obj_trans'][:, None] # 3x1
            rot_est = data_pred[k]['obj_rot']
            try:
                overts_pred = np.matmul(overts, rot_est.T) + data_pred[k]['obj_trans']
            except Exception as e:
                import pdb; pdb.set_trace()

            # occ_ratios = np.array(data_gt_params[seq]['occ_ratios'])
            # cam_K = config.ICAP_CAM_K if dname == 'icap' else config.BEHAVE_CAM_K
            cam_K = config.intrinsic_map[dname]
            mssd = pose_error.mssd_verts(overts_pred, rot_gt, trans_gt, overts, symms) * self.m2mm
            mspd = pose_error.mspd_verts(overts_pred, rot_gt, trans_gt, cam_K, overts, symms)
            re = pose_error.re_symm(rot_est, rot_gt, symms)
            dia = diameter*self.m2mm
            for key in ["MSSD", "MSPD", "RE", 'diameter']:
                if f'{key}_{dname}' not in error_dict:
                    error_dict[f'{key}_{dname}'] = []
            error_dict[f'MSSD_{dname}'].append(mssd)
            error_dict[f'MSPD_{dname}'].append(mspd)
            error_dict[f'RE_{dname}'].append(re)
            error_dict[f'diameter_{dname}'].append(dia)

            end = time.time()
            # self.logging(f'{seq} done, {end-seq_start:.4f} seconds.')
        self.logging(f'All images done, total {time.time()-start:.4f} seconds.')

        # compute recalls
        steps = np.arange(0.05, 0.51, 0.05)
        recalls_all = {}
        for dname in config.DATASET_NAMES:
            err_max = {
                "MSSD": np.array(error_dict[f'diameter_{dname}']) * 1.0,
                "MSPD": self.pe_max,
                "RE": self.re_max
            }
            recalls = {f'{k}': [] for k in err_max.keys()}
            for step in steps:
                for err in err_max.keys():
                    thres = err_max[err] * step
                    mask = np.array(error_dict[f'{err}_{dname}']) < thres
                    recall = np.sum(mask) / len(mask)
                    recalls[err].append(recall)
            recall_dict = {f'{k}_{dname}': np.mean(v) for k, v in recalls.items()}
            recall_dict[f"AR_all_{dname}"] = np.mean(np.array(list(recall_dict.values()), dtype=float))  # for final ranking
            recalls_all.update(**recall_dict)
        return recalls_all


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



