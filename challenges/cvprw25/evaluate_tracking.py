"""
evaluate tracking h+o methods
"""

import sys, os
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'program')) # path for remote server
sys.path.append(os.path.join(os.getcwd(), 'challenges'))
import numpy as np
import trimesh
from tqdm import tqdm

from lib import metrics as metric
from cvprw25.evaluate_joint import JointReconEvaluator


class JointTrackEvaluator(JointReconEvaluator):
    def eval(self, res_dir, gt_dir, outfile):
        """

        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return:
        """
        data_gt, data_pred = self.load_data(gt_dir, res_dir)
        data_complete = self.check_data(data_gt, data_pred)
        if data_complete:
            start = time.time()
            error_dict = self.compute_errors(data_gt, data_pred)
            end = time.time()
            self.logging(f'Evaluation done after {end-start:.4f} seconds')
        else:
            error_dict = {}
            for dname in ['behave', 'icap']:
                ed_i = {f'{k}_{dname}':np.inf for k in ["SMPL", 'obj']}
                error_dict.update(**ed_i)
        self.write_errors(outfile, error_dict)

    def check_data(self, dict_gt:dict, dict_pred:dict):
        if len(dict_pred.keys()) != len(dict_gt['annotations'].keys()):
            self.logging("Not enough sequences are reconstructed for evaluation! Exiting...")
            print("recon:", len(dict_pred.keys()), "GT:", len(dict_gt['annotations'].keys()))
            return False
        else:
            self.logging(f"Start evaluating {len(dict_pred.keys())} sequences.")
            return True

    def compute_errors(self, data_gt, data_pred):
        """

        :param data_gt:
        :param data_pred:
        :return:
        """
        errors_all = {}
        # manager = mp.Manager()
        # errors_all = manager.dict()
        jobs = []
        for k in tqdm(sorted(data_pred.keys())):
            if k not in data_gt['annotations']:
                self.logging(f'sequence id {k} not found in GT data!')
                continue
            self.eval_seq(data_gt, data_pred, errors_all, k)

        #     p = mp.Process(target=self.eval_seq, args=(data_gt, data_pred, errors_all, k))
        #     p.start()
        #     jobs.append(p)
        # for job in jobs:
        #     job.join()
            # self.eval_seq(data_gt, data_pred, errors_all, k)

        errors_avg = {k: np.mean(v) for k, v in errors_all.items()}
        return errors_avg

    def compute_accel_error(self, th_gt, th_pr):
        """
        compute acceleration error for translation
        return: (N, )
        """
        accel_gt = th_gt[:-2] - 2 * th_gt[1:-1] + th_gt[2:]
        accel_pr = th_pr[:-2] - 2 * th_pr[1:-1] + th_pr[2:]
        err = np.sqrt(np.sum((accel_gt - accel_pr) ** 2, -1)) * self.m2mm
        return err

    def eval_seq(self, data_gt, data_pred, errors_all, k):
        """
        evaluate one sequence specified by k
        :param data_gt:
        :param data_pred:
        :param errors_all: the result dictionary
        :param k:
        :return:
        """
        start = time.time()
        self.logging(f'start evaluating {k}...')
        dname, gender, obj_name = data_gt['annotations'][k]['meta']
        temp_faces = data_gt['templates'][obj_name]['faces']
        temp_verts = data_gt['templates'][obj_name]['verts']
        # temp_samples = data_gt['templates'][obj_name]['samples']

        # compute acceleration error of human and object
        th_gt, to_gt = data_gt["annotations"][k]['trans'], data_gt["annotations"][k]['obj_trans'] # GT human obj translation
        th_pr, to_pr = data_pred[k]['trans'], data_pred[k]['obj_trans']
        if len(th_gt) != len(th_pr):
            self.logging(f'The number of object predictions does not match GT! {len(th_gt)}!={len(th_pr)}')
            exit(-1)
        if len(to_gt) != len(to_pr):
            self.logging(f'The number of human predictions does not match GT! {len(to_gt)}!={len(to_pr)}')
            exit(-1)
        err_h = np.mean(self.compute_accel_error(th_gt, th_pr))
        err_o = np.mean(self.compute_accel_error(to_gt, to_pr))
        if f'acc-h_{dname}' not in errors_all:
            errors_all[f'acc-h_{dname}'] = []
        if f'acc-o_{dname}' not in errors_all:
            errors_all[f'acc-o_{dname}'] = []
        errors_all[f'acc-h_{dname}'].append(err_h)
        errors_all[f'acc-o_{dname}'].append(err_o)

        freq = 10 # instead of evaluate all frames which is expensive, only evaluate over some key frames

        # compute Object
        rot_pr, trans_pr = data_pred[k]['obj_rot'][::freq], data_pred[k]['obj_trans'][::freq]  # (T, 3, 3) and (T, 3)
        ov_pr = np.matmul(temp_verts[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        # pts_pr = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        ov_gt = np.matmul(temp_verts[None].repeat(len(rot_pr), 0),
                          data_gt["annotations"][k]['obj_rot'][::freq].transpose(0, 2, 1)) + \
                data_gt["annotations"][k]['obj_trans'][::freq][:, None]
        # pts_gt = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), data_gt["annotations"][k]['obj_rot'].transpose(0, 2, 1)) + \
        #             data_gt["annotations"][k]['obj_trans'][:, None]
        if len(ov_pr) != len(ov_gt):
            self.logging(f'the number of object vertices does not match! {len(ov_pr)}!={len(ov_gt)}, seq id={k}')
            exit(-1)
            # return
        # compute SMPL
        pose_pred = data_pred[k]['pose']
        if pose_pred.shape[-1] == 72:
            pose_pred = self.smpl2smplh_pose(pose_pred)
        model = self.smplh_male if gender == 'male' else self.smplh_female
        sv_pr = model.update(pose_pred[::freq], data_pred[k]['betas'][::freq], data_pred[k]['trans'][::freq])[0]
        sv_gt = model.update(data_gt['annotations'][k]['pose'][::freq], data_gt['annotations'][k]['betas'][::freq],
                             data_gt['annotations'][k]['trans'][::freq])[0]
        ee = time.time()
        # print('time to finish SMPL forward:', ee-start) # this is the most time consuming part, it can take up to 6s to finish 1500 frames
        if len(sv_pr) != len(sv_gt):
            self.logging(f'the number of SMPL vertices does not match! {len(sv_pr)}!={len(sv_gt)}, seq id={k}')
            exit(-1)
            # return
        # classify based on dataset
        if f'SMPL_{dname}' not in errors_all:
            errors_all[f'SMPL_{dname}'] = []
        if f'obj_{dname}' not in errors_all:
            errors_all[f'obj_{dname}'] = []
        L = len(sv_pr)
        time_window = 300//freq
        arot, atrans, ascale = None, None, None  # global alignment
        for i in range(0, L, 1): # cannot evaluate all frames since codalab server is too slow: this is not the bootleneck
            if arot is None or i % time_window == 0:
                # combine all vertices in this window and align
                bend = min(L, i + time_window)
                indices = np.arange(i, bend)
                # print(sv_gt.shape, ov_gt.shape, sv_pr.shape, ov_pr.shape)
                verts_clip_gt = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_gt, ov_gt]], 0)
                verts_clip_pr = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_pr, ov_pr]], 0)
                ss = time.time()
                _, arot, atrans, ascale = metric.compute_similarity_transform(verts_clip_pr, verts_clip_gt)
                ee = time.time()
                # print('time to align:',ee -ss) # around 0.137s to align

            # align
            ov_pr_i = (ascale * arot.dot(ov_pr[i].T) + atrans).T
            # ov_pr_i = (ascale*arot.dot(pts_pr[i].T) + atrans).T
            sv_pr_i = (ascale * arot.dot(sv_pr[i].T) + atrans).T
            # compute errors
            # err_obj, err_smpl = self.compute_joint_errors(ov_gt[i], ov_pr_i, sv_gt[i], sv_pr_i, temp_faces, model.faces)

            # compute v2v instead of cd: cd is too expensive to compute
            err_obj = np.mean(np.sqrt(np.sum((ov_pr_i - ov_gt[i]) ** 2, -1)))
            err_smpl = np.mean(np.sqrt(np.sum((sv_pr_i - sv_gt[i]) ** 2, -1)))

            errors_all[f'obj_{dname}'].append(err_obj * self.m2mm)
            errors_all[f'SMPL_{dname}'].append(err_smpl * self.m2mm) # this continuous access slows down multi process a lot!
        end = time.time()
        print(f'seq {k} ({len(th_gt)} frames) done after {end-start:.4f} seconds')


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

    evaluator = JointTrackEvaluator(truth_dir)
    evaluator.eval(submit_dir, truth_dir, output_filename)

