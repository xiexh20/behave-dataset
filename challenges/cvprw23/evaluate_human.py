"""
human reconstruction errors, all frames must have results

Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from challenges.lib.evaluate_base import BaseEvaluator
from challenges.lib import metrics as metric
from challenges.lib.SMPL import SMPL

# dirs = ['', '../', '../../', "program"]
# for d in dirs:
#     cmd = f'ls -l {d}'
#     print(f'executing {cmd}:')
#     os.system(cmd)


class HumanEvaluator(BaseEvaluator):
    def __init__(self, model_root):
        super(HumanEvaluator, self).__init__()
        self.smplh_male = SMPL('male', model_root)
        self.smplh_female = SMPL('female', model_root)

    def check_data(self, dict_gt:dict, dict_pred:dict):
        """
        check if data are consistent, i.e. the number of frames in each sequence matches
        :param dict_gt:
        :param dict_pred:
        :return:
        """
        for seq in dict_gt.keys():
            if seq not in dict_pred:
                self.logging(f"No prediction results for sequence {seq}!")
                return False

            # check number of frames
            L1, L2 = len(dict_gt[seq]['frames']), len(dict_pred[seq]['frames'])
            if L1 != L2:
                self.logging(f"Number of frames do not match! GT: {L1}, predicted: {L2}")
                return False
            # check submitted data
            keys = ['poses', 'betas', 'trans']
            for k in keys:
                if L1 != len(dict_pred[seq][k]):
                    self.logging(f"{k} data of seq {seq} incomplete, GT length={L1}, the submitted data length: {len(dict_pred[seq][k])}")
                    return False
        return True

    def eval(self, res_dir, gt_dir, outfile):
        """
        check data validity and then evaluate
        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return: write results to the outfile
        """
        data_gt, data_pred = self.load_data(gt_dir, res_dir)

        data_complete = self.check_data(data_gt['annotations'], data_pred)
        if data_complete:
            err_dict = self.compute_errors(data_gt, data_pred)
        else:
            err_dict = {
                "MPJPE": np.inf,
                "MPJPE_PA": np.inf,
                "PCK": np.inf,
                "AUC": np.inf,
                "MPJAE": np.inf,
                "MPJAE_PA": np.inf
            }

        self.write_errors(outfile, err_dict)

    def compute_errors(self, data_gt, data_pred):
        """
        compute the SMPL errors
        :param data_gt:
        :param data_pred:
        :return:
        """
        jtrs_gt, rots_gt, _ = self.compute_smpl_res(data_gt['annotations'])
        jtrs_pred, rots_pred, _ = self.compute_smpl_res(data_pred)
        assert jtrs_gt.shape == jtrs_pred.shape, f'the GT joints shape {jtrs_gt.shape} does not match the predicted joints shape: {jtrs_pred.shape}'
        assert rots_gt.shape == rots_pred.shape, f'the GT rotation matrix {rots_gt.shape} does not match the predicted rotation matrix {rots_pred.shape}'
        self.logging("Data preparation done, now start evaluation...")

        # now compute errors
        # Joint errors and procrustes matrices
        MPJPE_final, MPJPE_PA_final, errors_pck, mat_procs = metric.compute_errors(jtrs_pred * self.m2mm,
                                                                                   jtrs_gt * self.m2mm)
        # PCK value
        pck_final = metric.compute_pck(errors_pck, metric.PCK_THRESH) * 100.
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
        rots_pred_or = rots_pred[:, metric.SMPL_OR_JOINTS, :, :] # apply only to selected joints
        mats_pred_prc = np.matmul(mats_procs_exp, rots_pred_or)
        # Compute differences between the predicted matrices after procrustes and GT matrices
        error_rot_pa = np.degrees(metric.joint_angle_error(mats_pred_prc, rots_gt))
        # joint angle error without alignment
        error_rot = np.degrees(metric.joint_angle_error(rots_pred_or, rots_gt))

        err_dict = {
            "MPJPE": MPJPE_final,
            "MPJPE_PA": MPJPE_PA_final,
            "PCK": pck_final,
            "AUC": auc_final,
            "MPJAE": error_rot,
            "MPJAE_PA": error_rot_pa
        }
        return err_dict

    def compute_smpl_res(self, data):
        """
        compute SMPL joints and rotation matrices
        :param data:
        packed results for all sequences, a dict of:
        {
            seq_name: {
                poses: Tx156 or Tx72
                betas: Tx10
                trans: Tx3
                gender: male/female
            }
        }
        :return:
            SMPL joints position: Lx24x3, L is the total number of frames
            SMPL rotation matrices: Lx24x3x3
            SMPL verts: Lx6890x3
        """
        joints, rots = [], []
        smpl_verts = []
        for seq in sorted(data.keys()):
            d = data[seq]
            poses = d['poses']
            if poses.shape[-1] == 72:
                # add zero hand pose, the GT data uses SMPL-H model
                poses = self.smpl2smplh_pose(poses)
            model = self.smplh_male if d['gender'] == 'male' else self.smplh_female
            verts, jtrs, glb_rot = model.update(poses, d['betas'], d['trans'])
            jtrs_smpl = np.concatenate([jtrs[:, :23], jtrs[:, 37:38]], 1) # keep SMPL joints only
            glb_rot_smpl = np.concatenate([glb_rot[:, :23], glb_rot[:, 37:38]], 1) # keep SMPl joints only
            joints.append(jtrs_smpl)
            rots.append(glb_rot_smpl)
            smpl_verts.append(verts)
        return np.concatenate(joints, 0), np.concatenate(rots, 0), np.concatenate(smpl_verts, 0)




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
