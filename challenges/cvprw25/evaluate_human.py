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
        self.genders = ['male', 'female', 'neutral']
        self.body_models = {g:SMPL(g, model_root) for g in self.genders}
        # self.smplh_male = SMPL('male', model_root)
        # self.smplh_female = SMPL('female', model_root)

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
        keys = ['pose', 'betas', 'trans', 'joints', 'vertices'] #
        dataset_names = ['behave', 'icap', 'synz', 'imhd']

        pred_all = {x:{} for x in dataset_names}
        gt_all = {x: {} for x in dataset_names}
        # categorize all images based on dataset and gender
        for k, v in data_pred.items():
            if k not in data_gt['annotations']:
                self.logging(f'image id {k} not found in GT data!')
                continue
            dataset, gender = data_gt['annotations'][k]['meta'][:2]
            if gender not in pred_all[dataset]:
                pred_all[dataset][gender] = {x:[] for x in ['pose', 'betas', 'trans', 'id', 'joints', 'vertices']}
            if gender not in gt_all[dataset]:
                gt_all[dataset][gender] = {x:[] for x in ['pose', 'betas', 'trans', 'id']}
            for name in keys:
                if name in ['pose', 'betas', 'trans']:
                    # gt data always uses parameters
                    gt_all[dataset][gender][name].append(data_gt['annotations'][k][name].flatten())
                if name in data_pred[k]:
                    # prediction uses pose parameters or joints and vertices as well
                    pred_all[dataset][gender][name].append(data_pred[k][name])

        # evaluate each dataset separately
        errors_all = {}
        for dname in dataset_names:
            rots_gt_all, jtrs_gt_all = [], []
            rots_pred_all, jtrs_pred_all = [], []
            verts_pr, verts_gt = [], []
            for gender in self.genders:
                if gender not in pred_all[dname]:
                    continue
                if len(pred_all[dname][gender]['pose']) > 0 or len(pred_all[dname][gender]['joints']) > 0:
                    # compute GT and predicted SMPL
                    rots_gt, jtrs_gt, vs_gt = self.smpl_forward(dname, gender, gt_all)
                    if len(pred_all[dname][gender]['pose']) > 0:
                        # prefer to use pose parameters
                        rots_pred, jtrs_pred, vs_pr = self.smpl_forward(dname, gender, pred_all)
                    elif len(pred_all[dname][gender]['joints']) > 0:
                        # use joints and vertices
                        jtrs_pred, vs_pr = np.stack(pred_all[dname][gender]['joints'], 0), np.stack(pred_all[dname][gender]['vertices'], 0)
                        rots_pred = np.zeros_like(rots_gt) # not using rotation anymore
                    else:
                        continue

                    rots_gt_all.append(rots_gt.astype(float))
                    jtrs_gt_all.append(jtrs_gt.astype(float))
                    rots_pred_all.append(rots_pred.astype(float))
                    jtrs_pred_all.append(jtrs_pred.astype(float))
                    verts_pr.append(vs_pr.astype(float))
                    verts_gt.append(vs_gt.astype(float))
            if len(rots_gt_all) == 0:
                continue
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
        # model = self.smplh_male if gender == 'male' else self.smplh_female
        model = self.body_models[gender]
        pose = np.stack(pred_all[dname][gender]['pose'], 0)
        betas = np.stack(pred_all[dname][gender]['betas'], 0)
        trans = np.stack(pred_all[dname][gender]['trans'], 0)
        verts, jtrs, glb_rot = model.update(pose, betas, trans)
        jtrs_smpl = np.concatenate([jtrs[:, :23], jtrs[:, 37:38]], 1)  # keep SMPL joints only
        glb_rot_smpl = np.concatenate([glb_rot[:, :23], glb_rot[:, 37:38]], 1)  # keep SMPl joints only
        return glb_rot_smpl, jtrs_smpl, verts

    def compute_smpl_errors(self, jtrs_gt, jtrs_pred, rots_gt, rots_pred, verts_gt, verts_pr):
        """
        Update March24: not computing rotation errors anymore
        :param jtrs_gt: (N, J, 3)
        :param jtrs_pred: (N, J, 3)
        :param rots_gt: (N, J, 3, 3)
        :param rots_pred: (N, J, 3, 3)
        :param verts_pr: (N, V_s, 3)
        :param verts_gt: (N, V_s, 3)
        :return:
        """
        assert jtrs_gt.shape == jtrs_pred.shape, f'the given joint shape does not match: pred{jtrs_pred.shape}!=gt{jtrs_gt.shape}'
        assert rots_gt.shape == rots_pred.shape, f'the given rotation shape does not match: pred{rots_gt.shape}!=gt{rots_pred.shape}'
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
        # Compute orientation errors
        # Apply procrustus rotation to the global rotation matrices
        # mats_procs_exp = np.expand_dims(mat_procs, 1)
        # mats_procs_exp = np.tile(mats_procs_exp, (1, len(metric.SMPL_OR_JOINTS), 1, 1))
        # rots_pred_or = rots_pred[:, metric.SMPL_OR_JOINTS, :, :]  # apply only to selected joints
        # mats_pred_prc = np.matmul(mats_procs_exp, rots_pred_or)
        # # Compute differences between the predicted matrices after procrustes and GT matrices
        # error_rot_pa = np.degrees(metric.joint_angle_error(mats_pred_prc, rots_gt))
        # # Joint angle error without alignment
        # error_rot = np.degrees(metric.joint_angle_error(rots_pred_or, rots_gt))
        err_dict = {
            "MPJPE": MPJPE_final,
            "MPJPE_PA": MPJPE_PA_final,
            # "PCK": pck_final,
            "AUC": auc_final,
            # "MPJAE": 16.2, # still save MPJAE for legacy compatibility
            # "MPJAE_PA": 16.2,
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
