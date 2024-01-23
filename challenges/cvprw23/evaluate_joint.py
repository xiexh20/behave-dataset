"""
joint reconstruction evaluation, same evaluation metric used in CHORE

Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
import time

sys.path.append(os.getcwd())
import numpy as np
import trimesh
from tqdm import tqdm

from evaluate_human import HumanEvaluator
from challenges.lib import metrics as metric
from challenges.lib.chamfer_distance import chamfer_distance


class JointReconEvaluator(HumanEvaluator):
    def eval(self, res_dir, gt_dir, outfile):
        """
        do Procrustes alignment on combined mesh and then computer CD on SMPL and objects
        only evaluate frames with occlusion ratio < 0.3
        allow two type of submissions:
            1. SMPL + object templates
            2. SMPL + object vertices, the object vertices should match the released object templates
        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return:
        """
        data_gt, data_pred = self.load_data(gt_dir, res_dir)

        data_complete = self.check_data(data_gt['annotations'], data_pred)
        if data_complete:
            err_dict = self.computer_errors(data_gt, data_pred)
        else:
            err_dict = {
                "SMPL":np.inf,
                "Object":np.inf
            }
        self.write_errors(outfile, err_dict)

    def computer_errors(self, data_gt, data_pred):
        """
        compute SMPL and object Chamfer distance after combined alignment
        :param data_gt:
        :param data_pred:
        :return:
        """
        # prepare human and object vertices and faces
        data_gt_params = data_gt['annotations']
        # iterate over all sequences
        errors_smpl, errors_obj = [], []
        start = time.time()
        for seq in tqdm(sorted(data_gt_params.keys())):
            start_seq = time.time()
            occ_ratios = np.array(data_gt_params[seq]['occ_ratios'])
            # mask = occ_ratios >= self.occ_thres
            d = data_gt_params[seq]
            model = self.smplh_male if d['gender'] == 'male' else self.smplh_female
            sverts_gt, jtrs, glb_rot = model.update(d['poses'], d['betas'], d['trans'])

            # get estimated SMPL verts
            if 'smpl_verts' not in data_pred[seq]:
                # compute from parameters
                poses = data_pred[seq]['poses']
                if poses.shape[-1] == 72:
                    poses = self.smpl2smplh_pose(poses)
                sverts_pred, _, _ = model.update(poses, data_pred[seq]['betas'], data_pred[seq]['trans'])
            else:
                sverts_pred = data_pred[seq]['smpl_verts']

            # GT object
            obj_name = self.get_obj_name(seq)
            temp_faces = data_gt['templates'][obj_name]['faces']
            temp_verts = data_gt['templates'][obj_name]['verts']

            overts_gt = self.compute_overts_gt(data_gt_params, seq, temp_verts)
            if 'obj_verts' not in data_pred[seq]:
                overts_pred = self.compute_overts_pred(data_pred, seq, temp_verts)
            else:
                overts_pred = data_pred[seq]['obj_verts']

            # do alignment and evaluation
            frames_pred = data_pred[seq]['frames']
            for idx, frame in enumerate(data_gt_params[seq]['frames']):
                if occ_ratios[idx] < self.occ_thres:
                    continue
                if frame not in frames_pred:
                    continue
                pidx = frames_pred.index(frame) # find the recon index and get the data
                sv_gt, ov_gt = sverts_gt[idx], overts_gt[idx]
                sv_pr, ov_pr = sverts_pred[pidx],overts_pred[pidx]
                # print(sv_pr.shape, ov_pr.shape)

                V = sv_gt.shape[0]
                st = time.time()
                aligned, R, t, scale = metric.compute_similarity_transform(np.concatenate([sv_pr, ov_pr]),
                                                                     np.concatenate([sv_gt, ov_gt]))
                sv_pr, ov_pr = aligned[:V], aligned[V:]
                # compute errors
                smpl_samples = [self.surface_sampling(v, model.faces) for v in [sv_gt, sv_pr]]
                obj_samples = [self.surface_sampling(v, temp_faces) for v in [ov_gt, ov_pr]]
                st = time.time()
                err_smpl = chamfer_distance(smpl_samples[0], smpl_samples[1])
                # print('CD human time: {}', time.time() - st) # 0.06-0.07s/example for 10k samples
                st = time.time()
                err_obj = chamfer_distance(obj_samples[0], obj_samples[1])
                # print('CD Object time: {}', time.time() - st)

                errors_smpl.append(err_smpl*self.m2mm)
                errors_obj.append(err_obj*self.m2mm)
            end = time.time()
            self.logging(f'{seq} done, {end - start_seq:.4f} seconds.')
        self.logging(f'All sequences done, total {time.time() - start:.4f} seconds.')
        err_dict = {
            "SMPL": np.mean(errors_smpl),
            "Object": np.mean(errors_obj)
        }

        return err_dict

    def surface_sampling(self, verts, faces):
        "sample points on the surface"
        m = self.to_trimesh(verts, faces)
        points = m.sample(self.sample_num)
        return points

    def to_trimesh(self, verts, faces):
        "psbody mesh to trimesh"
        trim = trimesh.Trimesh(verts, faces, process=False)
        return trim

    def check_data(self, dict_gt:dict, dict_pred:dict):
        """
        check smpl and object parameters/vertices
        :param dict_gt:
        :param dict_pred:
        :return:
        """
        counts_gt, counts_pred = 0, 0
        for seq in dict_gt.keys():
            # check number of frames based on occlusion ratio
            L1 = np.sum(np.array(dict_gt[seq]['occ_ratios'])>=self.occ_thres)
            # L2 = len(dict_pred[seq]['frames'])
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
        return True


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