"""
evaluate joint reconstruction methods
"""
import sys, os
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'program'))
sys.path.append(os.path.join(os.getcwd(), 'challenges'))
import numpy as np
import trimesh
from tqdm import tqdm


from lib import metrics as metric
from lib.chamfer_distance import chamfer_distance
from cvprw24.evaluate_human import HumanEvaluator
import lib.config as config


class JointReconEvaluator(HumanEvaluator):
    def eval(self, res_dir, gt_dir, outfile):
        """
        accept input as SMPL pose, shape parameters and object pose parameters.
        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return:
        """
        data_gt, data_pred = self.load_data(gt_dir, res_dir)
        data_complete = self.check_data(data_gt, data_pred)
        if data_complete:
            error_dict = self.compute_errors(data_gt, data_pred)
        else:
            error_dict = {}
            for dname in config.DATASET_NAMES:
                ed_i = {f'{k}_{dname}':np.inf for k in ["SMPL", 'obj']}
                error_dict.update(**ed_i)
        self.write_errors(outfile, error_dict)

    def compute_errors(self, data_gt, data_pred):
        """

        :param data_gt:
        :param data_pred:
        :return:
        """
        errors_all = {}
        for k in tqdm(sorted(data_pred.keys())):
            if k not in data_gt['annotations']:
                self.logging(f'image id {k} not found in GT data!')
                continue
            # continue
            dname, gender, obj_name = data_gt['annotations'][k]['meta']
            temp_faces = data_gt['templates'][obj_name]['faces']
            temp_verts = data_gt['templates'][obj_name]['verts']

            # compute Object
            ov_pr = np.matmul(temp_verts, data_pred[k]['obj_rot'].T) + data_pred[k]['obj_trans']
            ov_gt = np.matmul(temp_verts, data_gt["annotations"][k]['obj_rot'].T) + data_gt["annotations"][k]['obj_trans']

            # compute SMPL
            model = self.smplh_male if gender == 'male' else self.smplh_female
            if 'vertices' in data_pred[k]:
                # use precomputed SMPL vertices
                sv_pr = data_pred[k]['vertices'].astype(float)
            else:
                pose_pred = data_pred[k]['pose']
                if pose_pred.shape[-1] == 72:
                    pose_pred = self.smpl2smplh_pose(pose_pred)
                sv_pr = model.update(pose_pred[None], data_pred[k]['betas'][None], data_pred[k]['trans'][None])[0][0]
            sv_gt = model.update(data_gt['annotations'][k]['pose'][None], data_gt['annotations'][k]['betas'][None],
                                     data_gt['annotations'][k]['trans'].flatten()[None])[0][0]
            assert sv_gt.shape == sv_pr.shape, f'the given SMPL shape {sv_pr.shape} does not match GT SMPL shape {sv_gt.shape}!'
            V = sv_gt.shape[0]
            # print(sv_gt.shape, ov_pr.shape, sv_pr.shape, ov_gt.shape, data_gt['annotations'][k]['pose'].shape,
            #       data_gt['annotations'][k]['betas'].shape, data_gt['annotations'][k]['trans'].shape)
            aligned, R, t, scale = metric.compute_similarity_transform(np.concatenate([sv_pr, ov_pr]),
                                                                       np.concatenate([sv_gt, ov_gt]))
            sv_pr, ov_pr = aligned[:V], aligned[V:]
            # compute errors
            err_obj, err_smpl = self.compute_joint_errors(ov_gt, ov_pr, sv_gt, sv_pr, temp_faces, model.faces)

            # classify based on dataset
            if f'SMPL_{dname}' not in errors_all:
                errors_all[f'SMPL_{dname}'] = []
            if f'obj_{dname}' not in errors_all:
                errors_all[f'obj_{dname}'] = []
            errors_all[f'obj_{dname}'].append(err_obj*self.m2mm)
            errors_all[f'SMPL_{dname}'].append(err_smpl*self.m2mm)

        errors_avg = {k:np.mean(v) for k, v in errors_all.items()}
        return errors_avg

    def compute_joint_errors(self, ov_gt, ov_pr, sv_gt, sv_pr, obj_faces, smpl_faces):
        """

        :param ov_gt: (N_o, 3)
        :param ov_pr: (N_o, 3)
        :param sv_gt: (N_s, 3)
        :param sv_pr: (N_s, 3)
        :param obj_faces: (F_o, 3)
        :param smpl_faces: (F_o, 3)
        :return:
        """
        smpl_samples = [self.surface_sampling(v, smpl_faces) for v in [sv_gt, sv_pr]]
        obj_samples = [self.surface_sampling(v, obj_faces) for v in [ov_gt, ov_pr]]
        err_smpl = chamfer_distance(smpl_samples[0], smpl_samples[1])
        err_obj = chamfer_distance(obj_samples[0], obj_samples[1])
        return err_obj, err_smpl

    def surface_sampling(self, verts, faces):
        "sample points on the surface"
        m = self.to_trimesh(verts, faces)
        points = m.sample(self.sample_num)
        return points

    def to_trimesh(self, verts, faces):
        "psbody mesh to trimesh"
        trim = trimesh.Trimesh(verts, faces, process=False)
        return trim



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

    evaluator = JointReconEvaluator(truth_dir)
    evaluator.eval(submit_dir, truth_dir, output_filename)