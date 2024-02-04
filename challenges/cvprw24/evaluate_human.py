import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'program'))
sys.path.append(os.path.join(os.getcwd(), 'challenges'))
import numpy as np
from lib.evaluate_base import BaseEvaluator
from lib import metrics as metric
from lib.SMPL import SMPL
import lib.config as config

class HumanEvaluator(BaseEvaluator):
    def __init__(self, model_root):
        super(HumanEvaluator, self).__init__()
        self.smplh_male = SMPL('male', model_root)
        self.smplh_female = SMPL('female', model_root)

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

    def eval(self, res_dir, gt_dir, outfile):
        """
        check data validity and then evaluate
        :param res_dir: contain file results.pkl
        :param gt_dir: contain file ref.pkl
        :param outfile:
        :return: write results to the outfile
        """
        data_gt, data_pred = self.load_data(gt_dir, res_dir)
        # check if all images are predicted
        data_complete = self.check_data(data_gt, data_pred)

        if not data_complete:
            error_dict = {}
            for dname in config.DATASET_NAMES:
                ed_i = {f'{k}_{dname}': np.inf for k in ["SMPL", 'obj']}
                error_dict.update(**ed_i)
            self.write_errors(outfile, error_dict)
            return

        # metadata = data_gt['metadata'] # a dict of imageid to gender and dataset name map
        keys = ['pose', 'betas', 'trans']
        dataset_names = ['behave', 'icap', 'synz']

        # gt data: {image_id: {'pose': ; 'betas':, 'trans': , }, }
        pred_all = {x:{} for x in dataset_names}
        gt_all = {x: {} for x in dataset_names}
        # categorize all images based on dataset and gender
        for k, v in data_pred.items():
            if k not in data_gt['annotations']:
                self.logging(f'image id {k} not found in GT data!')
                continue
            dataset, gender = data_gt['annotations'][k]['meta'][:2]
            if gender not in pred_all[dataset]:
                pred_all[dataset][gender] = {x:[] for x in ['pose', 'betas', 'trans', 'id']}
            if gender not in gt_all[dataset]:
                gt_all[dataset][gender] = {x:[] for x in ['pose', 'betas', 'trans', 'id']}
            for name in keys:
                pred_all[dataset][gender][name].append(data_pred[k][name].flatten())
                gt_all[dataset][gender][name].append(data_gt['annotations'][k][name].flatten())

        # evaluate each dataset separately
        errors_all = {}
        for dname in dataset_names:
            rots_gt_all, jtrs_gt_all = [], []
            rots_pred_all, jtrs_pred_all = [], []
            verts_pr, verts_gt = [], []
            for gender in ['male', 'female']:
                if gender not in pred_all[dname]:
                    continue
                if len(pred_all[dname][gender]['pose']) > 0:
                    # compute GT and predicted SMPL
                    rots_pred, jtrs_pred, vs_pr = self.smpl_forward(dname, gender, pred_all)
                    rots_gt, jtrs_gt, vs_gt = self.smpl_forward(dname, gender, gt_all)

                    rots_gt_all.append(rots_gt)
                    jtrs_gt_all.append(jtrs_gt)
                    rots_pred_all.append(rots_pred)
                    jtrs_pred_all.append(jtrs_pred)
                    verts_pr.append(vs_pr)
                    verts_gt.append(vs_gt)
            rots_gt, jtrs_gt = np.concatenate(rots_gt_all, 0), np.concatenate(jtrs_gt_all, 0)
            rots_pred, jtrs_pred = np.concatenate(rots_pred_all, 0), np.concatenate(jtrs_pred_all, 0)

            # compute errors
            errors = self.compute_smpl_errors(jtrs_gt, jtrs_pred, rots_gt, rots_pred, np.concatenate(verts_gt, 0),
                                              np.concatenate(verts_pr, 0))
            for k, v in errors.items():
                errors_all[f'{k}_{dname}'] = v
            # errors_all[dname] = errors
            self.logging(f'{dname} dataset evaluation done.')

        self.write_errors(outfile, errors_all)

    def smpl_forward(self, dname, gender, pred_all):
        model = self.smplh_male if gender == 'male' else self.smplh_female
        pose = np.stack(pred_all[dname][gender]['pose'], 0)
        betas = np.stack(pred_all[dname][gender]['betas'], 0)
        trans = np.stack(pred_all[dname][gender]['trans'], 0)
        verts, jtrs, glb_rot = model.update(pose, betas, trans)
        jtrs_smpl = np.concatenate([jtrs[:, :23], jtrs[:, 37:38]], 1)  # keep SMPL joints only
        glb_rot_smpl = np.concatenate([glb_rot[:, :23], glb_rot[:, 37:38]], 1)  # keep SMPl joints only
        return glb_rot_smpl, jtrs_smpl, verts

    def compute_smpl_errors(self, jtrs_gt, jtrs_pred, rots_gt, rots_pred, verts_gt, verts_pr):
        """

        :param jtrs_gt: (N, J, 3)
        :param jtrs_pred: (N, J, 3)
        :param rots_gt: (N, J, 3, 3)
        :param rots_pred: (N, J, 3, 3)
        :param verts_pr: (N, V_s, 3)
        :param verts_gt: (N, V_s, 3)
        :return:
        """
        assert jtrs_gt.shape == jtrs_pred.shape, f'the given joint shape does not match: {jtrs_pred.shape}!={jtrs_gt.shape}'
        assert rots_gt.shape == rots_pred.shape, f'the given rotation shape does not match: {rots_gt.shape}!={rots_pred.shape}'
        # make all verts root relative
        verts_gt, verts_pr = verts_gt - jtrs_gt[:, 0:1], verts_pr - jtrs_pred[:, 0:1]

        # Joint errors and procrustes matrices
        MPJPE_final, MPJPE_PA_final, errors_pck, mat_procs = metric.compute_errors(jtrs_pred * self.m2mm,
                                                                                   jtrs_gt * self.m2mm)
        # Procrustes alignment for vertices
        verts_pr_proc = np.matmul(verts_pr, mat_procs.transpose(0, 2, 1))
        # V2V of SMPL vertices
        v2v = np.mean(np.sqrt(np.sum((verts_pr_proc - verts_gt)**2, -1)))

        # PCK value
        # pck_final = metric.compute_pck(errors_pck, metric.PCK_THRESH) * 100.
        # AUC value
        auc_range = np.arange(metric.AUC_MIN, metric.AUC_MAX)
        pck_aucs = []
        for pck_thresh_ in auc_range:
            err_pck_tem = metric.compute_pck(errors_pck, pck_thresh_)
            pck_aucs.append(err_pck_tem)
        auc_final = metric.compute_auc(auc_range / auc_range.max(), pck_aucs)
        # compute orientation errors
        # Apply procrustus rotation to the global rotation matrices
        mats_procs_exp = np.expand_dims(mat_procs, 1)
        mats_procs_exp = np.tile(mats_procs_exp, (1, len(metric.SMPL_OR_JOINTS), 1, 1))
        rots_pred_or = rots_pred[:, metric.SMPL_OR_JOINTS, :, :]  # apply only to selected joints
        mats_pred_prc = np.matmul(mats_procs_exp, rots_pred_or)
        # Compute differences between the predicted matrices after procrustes and GT matrices
        error_rot_pa = np.degrees(metric.joint_angle_error(mats_pred_prc, rots_gt))
        # joint angle error without alignment
        error_rot = np.degrees(metric.joint_angle_error(rots_pred_or, rots_gt))
        err_dict = {
            "MPJPE": MPJPE_final,
            "MPJPE_PA": MPJPE_PA_final,
            # "PCK": pck_final,
            "AUC": auc_final,
            "MPJAE": error_rot,
            "MPJAE_PA": error_rot_pa,
            'v2v': v2v*self.m2mm
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

    evaluator = HumanEvaluator(truth_dir)
    evaluator.eval(submit_dir, truth_dir, output_filename)
