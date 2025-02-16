"""
multi-processing for track evaluation
"""
import copy
import sys, os
import time

sys.path.append(os.getcwd())
import numpy as np
import trimesh
from tqdm import tqdm
from copy import deepcopy

from lib import metrics as metric
from lib.chamfer_distance import chamfer_distance
from cvprw25.evaluate_tracking import JointTrackEvaluator
import lib.config as config

import multiprocessing as mp

class JointTrackEvaluatorMP(JointTrackEvaluator):
    def compute_errors(self, data_gt, data_pred):
        """
        use multi-process
        :param data_gt:
        :param data_pred:
        :return:
        """

        manager = mp.Manager()
        errors_all = manager.dict()
        jobs = []
        for k in sorted(data_pred.keys()):
            if k not in data_gt['annotations']:
                self.logging(f'sequence id {k} not found in GT data!')
                continue
            # self.eval_seq(data_gt, data_pred, errors_all, k)
            # dname, gender, obj_name = data_gt['annotations'][k]['meta']
            # temp_verts = data_gt['templates'][obj_name]['verts'].copy()
            # p = mp.Process(target=self.eval_seq_opt, args=(deepcopy(data_gt['annotations'][k]),
            #                                            deepcopy(data_pred[k]), errors_all, k, temp_verts))
            p = mp.Process(target=self.eval_seq, args=(data_gt, data_pred, errors_all, k))
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join()

        # collect results
        errors_ret = {}
        for k, v in errors_all.items():
            dname, gender, obj_name = data_gt['annotations'][k]['meta']
            esmpl, eobj = v
            if f'SMPL_{dname}' not in errors_ret:
                errors_ret[f'SMPL_{dname}'] = []
            if f'obj_{dname}' not in errors_all:
                errors_ret[f'obj_{dname}'] = []
            errors_ret[f'SMPL_{dname}'].extend(esmpl)
            errors_ret[f'obj_{dname}'].extend(eobj)

        errors_avg = {k: np.mean(v) for k, v in errors_ret.items()}
        return errors_avg

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

        # compute Object
        rot_pr, trans_pr = data_pred[k]['obj_rot'], data_pred[k]['obj_trans']  # (T, 3, 3) and (T, 3)
        ov_pr = np.matmul(temp_verts[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        # pts_pr = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        ov_gt = np.matmul(temp_verts[None].repeat(len(rot_pr), 0),
                          data_gt["annotations"][k]['obj_rot'].transpose(0, 2, 1)) + \
                data_gt["annotations"][k]['obj_trans'][:, None]
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
        sv_pr = model.update(pose_pred, data_pred[k]['betas'], data_pred[k]['trans'])[0]
        sv_gt = model.update(data_gt['annotations'][k]['pose'], data_gt['annotations'][k]['betas'],
                             data_gt['annotations'][k]['trans'])[0]
        if len(sv_pr) != len(sv_gt):
            self.logging(f'the number of SMPL vertices does not match! {len(sv_pr)}!={len(sv_gt)}, seq id={k}')
            exit(-1)
            # return
        # classify based on dataset
        # if f'SMPL_{dname}' not in errors_all:
        #     errors_all[f'SMPL_{dname}'] = []
        # if f'obj_{dname}' not in errors_all:
        #     errors_all[f'obj_{dname}'] = []
        errs_smpl, errs_obj = [], []
        L = len(sv_pr)
        time_window = 300
        arot, atrans, ascale = None, None, None  # global alignment
        for i in tqdm(range(L)):
            if arot is None or i % time_window == 0:
                # combine all vertices in this window and align
                bend = min(L, i + time_window)
                indices = np.arange(i, bend)
                # print(sv_gt.shape, ov_gt.shape, sv_pr.shape, ov_pr.shape)
                verts_clip_gt = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_gt, ov_gt]], 0)
                verts_clip_pr = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_pr, ov_pr]], 0)
                _, arot, atrans, ascale = metric.compute_similarity_transform(verts_clip_pr, verts_clip_gt)

            # align
            ov_pr_i = (ascale * arot.dot(ov_pr[i].T) + atrans).T
            # ov_pr_i = (ascale*arot.dot(pts_pr[i].T) + atrans).T
            sv_pr_i = (ascale * arot.dot(sv_pr[i].T) + atrans).T
            # compute errors
            # err_obj, err_smpl = self.compute_joint_errors(ov_gt[i], ov_pr_i, sv_gt[i], sv_pr_i, temp_faces, model.faces)

            # compute v2v instead of cd: cd is too expensive to compute
            err_obj = np.mean(np.sqrt(np.sum((ov_pr_i - ov_gt[i]) ** 2, -1)))
            err_smpl = np.mean(np.sqrt(np.sum((sv_pr_i - sv_gt[i]) ** 2, -1)))
            errs_obj.append(err_obj * self.m2mm)
            errs_smpl.append(err_smpl * self.m2mm)

            # errors_all[f'obj_{dname}'].append(err_obj * self.m2mm)
            # errors_all[f'SMPL_{dname}'].append(err_smpl * self.m2mm) # this continuous access slows down multi process a lot!
        end = time.time()
        print(f'seq {k} done after {end-start:.4f} seconds')



    def eval_seq_opt(self, data_gt, data_pred, errors_all, k, temp_verts):
        "optimized for multi-processing, this is even slower"
        start = time.time()
        # self.logging(f'start evaluating {k}...')
        dname, gender, obj_name = data_gt['meta'] # this is also slow!
        # temp_faces = data_gt['templates'][obj_name]['faces']
        # temp_verts = data_gt['templates'][obj_name]['verts']
        # temp_samples = data_gt['templates'][obj_name]['samples']
        # compute Object
        rot_pr, trans_pr = data_pred['obj_rot'], data_pred['obj_trans']  # (T, 3, 3) and (T, 3)
        ov_pr = np.matmul(temp_verts[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        # pts_pr = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        ov_gt = np.matmul(temp_verts[None].repeat(len(rot_pr), 0),
                          data_gt['obj_rot'].transpose(0, 2, 1)) + \
                data_gt['obj_trans'][:, None]
        # pts_gt = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), data_gt['obj_rot'].transpose(0, 2, 1)) + \
        #             data_gt['obj_trans'][:, None]
        if len(ov_pr) != len(ov_gt):
            self.logging(f'the number of object vertices does not match! {len(ov_pr)}!={len(ov_gt)}, seq id={k}')
            exit(-1)
            # return
        # compute SMPL
        pose_pred = data_pred['pose']
        if pose_pred.shape[-1] == 72:
            pose_pred = self.smpl2smplh_pose(pose_pred)
        model = self.smplh_male if gender == 'male' else self.smplh_female
        sv_pr = model.update(pose_pred, data_pred['betas'], data_pred['trans'])[0]
        sv_gt = model.update(data_gt['pose'], data_gt['betas'],
                             data_gt['trans'])[0]
        if len(sv_pr) != len(sv_gt):
            self.logging(f'the number of SMPL vertices does not match! {len(sv_pr)}!={len(sv_gt)}, seq id={k}')
            exit(-1)
            # return
        # classify based on dataset
        # if f'SMPL_{dname}' not in errors_all:
        #     errors_all[f'SMPL_{dname}'] = []
        # if f'obj_{dname}' not in errors_all:
        #     errors_all[f'obj_{dname}'] = []
        errs_smpl, errs_obj = [], []
        L = len(sv_pr)
        time_window = 300
        arot, atrans, ascale = None, None, None  # global alignment
        self.logging(f'start evaluating {k}...')
        for i in tqdm(range(L)):
            if arot is None or i % time_window == 0:
                # combine all vertices in this window and align
                bend = min(L, i + time_window)
                indices = np.arange(i, bend)
                # print(sv_gt.shape, ov_gt.shape, sv_pr.shape, ov_pr.shape)
                verts_clip_gt = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_gt, ov_gt]], 0)
                verts_clip_pr = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_pr, ov_pr]], 0)
                _, arot, atrans, ascale = metric.compute_similarity_transform(verts_clip_pr, verts_clip_gt)

            # align
            ov_pr_i = (ascale * arot.dot(ov_pr[i].T) + atrans).T
            # ov_pr_i = (ascale*arot.dot(pts_pr[i].T) + atrans).T
            sv_pr_i = (ascale * arot.dot(sv_pr[i].T) + atrans).T
            # compute errors
            # err_obj, err_smpl = self.compute_joint_errors(ov_gt[i], ov_pr_i, sv_gt[i], sv_pr_i, temp_faces, model.faces)

            # compute v2v instead of cd: cd is too expensive to compute
            err_obj = np.mean(np.sqrt(np.sum((ov_pr_i - ov_gt[i]) ** 2, -1)))
            err_smpl = np.mean(np.sqrt(np.sum((sv_pr_i - sv_gt[i]) ** 2, -1)))

            # errors_all[f'obj_{dname}'].append(err_obj * self.m2mm)
            # errors_all[f'SMPL_{dname}'].append(
            #     err_smpl * self.m2mm)  # this continuous access slows down multi process a lot!
            errs_obj.append(err_obj*self.m2mm)
            errs_smpl.append(err_smpl*self.m2mm)
        end = time.time()
        errors_all[k] = (errs_smpl, errs_obj)
        print(f'seq {k} done after {end - start:.4f} seconds')


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

    evaluator = JointTrackEvaluatorMP(truth_dir)
    evaluator.eval(submit_dir, truth_dir, output_filename)

