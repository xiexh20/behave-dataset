"""
simple script to demonstrate converting smpl params to SMPL mesh
"""
import sys, os
sys.path.append(os.getcwd())
import pickle as pkl
import numpy as np
import trimesh
import torch
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer

seq_root = "ROOT" # replace this with the root path of behave sequences
frame = "Date01_Sub01_chairwood_hand/t0003.000"
param_file = f"{seq_root}/{frame}/person/fit02/person_fit.pkl"
mesh_file = f"{seq_root}/{frame}/person/fit02/person_fit.ply"
gender = 'male' # gender information can be found in SEQ_PATH/info.json

# path to SMPL-H model (MANOv1.2), see https://github.com/bharat-b7/MPI_MeshRegistration/tree/main/smpl_registration#smplh-files
model_root = "MODEL_ROOT"

smpl_dict = pkl.load(open(param_file, 'rb'))
p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
pose = torch.tensor([p])
betas = torch.tensor([b])
trans = torch.tensor([t])

smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=10,
                               model_root=str(model_root), hands=True)
verts, _, _, _ = smpl(pose, th_betas=betas, th_trans=trans)
verts = verts[0].cpu().numpy()
faces = smpl.th_faces.cpu().numpy()

mesh = trimesh.load_mesh(mesh_file, process=False)

err = np.sum((mesh.vertices-verts)**2)
assert err < 1e-8, f'err={err}'
print('all done')


