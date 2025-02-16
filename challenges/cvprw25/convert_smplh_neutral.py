"""
simple script to convert the SMPLH model stored in npz file to pkl file
"""
import pickle as pkl
import numpy as np
from scipy.sparse import coo_matrix


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('npz_path', type=str)
    parser.add_argument('pkl_path', type=str)

    args = parser.parse_args()

    smpl_data = np.load(args.npz_path, allow_pickle=True)
    # do some conversion to be compatible with our SMPL model code
    new_dict = {k:smpl_data[k] for k in smpl_data}
    new_dict['J_regressor'] = coo_matrix(new_dict['J_regressor'])
    new_dict['shapedirs'] = new_dict['shapedirs'][:, :, :10] # keep only the first 10 betas
    pkl.dump(new_dict, open(args.pkl_path, 'wb'))

    print(f'all done, saved to {args.pkl_path}')

if __name__ == '__main__':
    main()