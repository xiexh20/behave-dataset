"""
template free joint human and object reconstruction

metrics: F-score of human, object and combined point cloud
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
from lib.chamfer_distance import chamfer_distance, compute_fscore
from cvprw25.evaluate_joint import JointReconEvaluator
import lib.config as config


class TFreeJointReconEvaluator(JointReconEvaluator):
    def get_error_keys(self):
        return ["Hum", 'Obj', 'Comb']

    def compute_errors(self, data_gt, data_pred):
        """

        :param data_gt:
        :param data_pred:
        :return: a dict of errors
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

            # prepare GT meshes
            ov_gt = np.matmul(temp_verts, data_gt["annotations"][k]['obj_rot'].T) + data_gt["annotations"][k]['obj_trans']
            model = self.smplh_male if gender == 'male' else self.smplh_female
            sv_gt = model.update(data_gt['annotations'][k]['pose'][None], data_gt['annotations'][k]['betas'][None],
                                 data_gt['annotations'][k]['trans'].flatten()[None])[0][0]
            samples_hum_gt = self.surface_sampling(sv_gt, model.faces)
            samples_obj_gt = self.surface_sampling(ov_gt, temp_faces)
            # normalize points
            samples = np.concatenate([samples_hum_gt, samples_obj_gt], axis=0)
            cent = np.mean(samples, axis=0)
            radius = np.sqrt(np.max(np.sum((samples - cent) ** 2, -1)))
            samples = (samples - cent)/ (radius*2)
            samples_hum_gt, samples_obj_gt = samples[:self.sample_num], samples[self.sample_num:]

            samples_hum_pr = data_pred[k]['hum_pts'] # Predicted human points
            samples_obj_pr = data_pred[k]['obj_pts'] # Predicted object points
            if len(samples_hum_gt) != len(samples_hum_pr):
                sample_indices = np.random.choice(len(samples_hum_pr), size=self.sample_num, replace=len(samples_hum_pr)<len(samples_hum_gt))
                samples_hum_pr = samples_hum_pr[sample_indices]
            if len(samples_obj_gt) != len(samples_obj_pr):
                sample_indices = np.random.choice(len(samples_obj_pr), size=self.sample_num, replace=len(samples_obj_pr)<len(samples_obj_gt))
                samples_obj_pr = samples_obj_pr[sample_indices]
            # compute numbers
            score_hum = compute_fscore(samples_hum_gt, samples_hum_pr)[0]
            score_obj = compute_fscore(samples_obj_gt, samples_obj_pr)[0]
            score_comb = compute_fscore(np.concatenate([samples_hum_gt, samples_obj_gt], axis=0),
                                        np.concatenate([samples_hum_pr, samples_obj_pr], axis=0))[0]

            # for debug
            # trimesh.PointCloud(np.concatenate([samples_hum_gt, samples_obj_gt], axis=0)).export(f'debug/{k}_gt.ply')
            # trimesh.PointCloud(np.concatenate([samples_hum_pr, samples_obj_pr], axis=0)).export(f'debug/{k}_pr.ply')
            # print(f'{k} human: {score_hum}, obj: {score_obj}, comb: {score_comb}')
            # break

            err_keys = self.get_error_keys()
            for k in err_keys:
                if f'{k}_{dname}' not in errors_all:
                    errors_all[f'{k}_{dname}'] = []
            for k, score in zip(err_keys, [score_hum, score_obj, score_comb]):
                errors_all[f'{k}_{dname}'].append(score)

        errors_avg = {k: np.mean(v) for k, v in errors_all.items()}
        return errors_avg




if __name__ == '__main__':
    # Default execution for codalab
    # Example usage: python cvprw25/evaluate_joint_tfree.py submit/cvprw25/tfree submit/cvprw25/tfree/scores.txt
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Process reference and results directory
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, 'scores.txt')

    evaluator = TFreeJointReconEvaluator(truth_dir)
    evaluator.eval(submit_dir, truth_dir, output_filename)