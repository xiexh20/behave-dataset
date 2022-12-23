import numpy as np

object_dimensions ={
    'person': [1.5, 2.5, 1.5], # x, y, z dimension, in meter
    "backpack":[0.8, 0.8, 0.8],
    'basketball':[0.5, 0.5, 0.5],
    'boxlarge':[0.8, 0.8, 0.8],
    'boxlong':[1.0, 1.0, 1.0],
    'boxmedium':[0.7, 0.7, 0.7],
    'boxsmall':[0.5, 0.5, 0.5],
    'boxtiny':[0.5, 0.5, 0.5],
'chair': [1.0, 1.0, 1.0],
'chairwood': [1.0, 1.0, 1.0],
'chairblack': [1.0, 1.0, 1.0],
'keyboard':[0.8, 0.8, 0.8],
'monitor':[0.8, 0.8, 0.8],
'plasticcontainer':[0.8, 0.8, 0.8],
'stool':[0.8, 0.8, 0.8],
'suitcase':[0.8, 0.8, 0.8],
'table':[1.2, 1.2, 1.2],
'tablesmall':[1.2, 1.2, 1.2],
'tablesquare':[1.2, 1.2, 1.2],
'toolbox':[0.5, 0.5, 0.5],
'trashbin':[0.6, 0.6, 0.6],
'yogaball':[1.0, 1., 1.],
    'sports ball': [1.0, 1., 1.],
    'yogamat':[1.0, 1., 1.],

    "shapenet":[1.5, 1.5, 1.5] # synthetic data
}


class PCloudsFilter:
    def __init__(self):
        pass

    @staticmethod
    def nobkg_3dmask(center_point, camera_points, label):
        """
        the mask to remove background
        center_point: (3, )
        camera_points: (N, 3)
        """
        box_min = center_point - np.array(object_dimensions[label]) / 2.0
        box_max = center_point + np.array(object_dimensions[label]) / 2.0
        validx_idx = np.logical_and(camera_points[:, 0] <= box_max[0], camera_points[:, 0] >= box_min[0]) # x mask, (N, )
        validy_idx = np.logical_and(camera_points[:, 1] <= box_max[1], camera_points[:, 1] >= box_min[1]) # y mask, (N, )
        validz_idx = np.logical_and(camera_points[:, 2] <= box_max[2], camera_points[:, 2] >= box_min[2]) # z mask
        valid_idx = np.logical_and(np.logical_and(validx_idx, validy_idx), validz_idx) # combined mask

        return valid_idx

    @staticmethod
    def filter_pclouds(pc, pc_color, label):
        # filter out background points using median center
        median = np.median(pc, 0)
        # print(median)
        valid_idx = PCloudsFilter.nobkg_3dmask(median, pc, label)
        if np.sum(valid_idx) ==0:
            print("{} filter out completely".format(label))
            return None, None
        pc = pc[valid_idx, :]
        pc_color = pc_color[valid_idx, :]

        return pc, pc_color

    @staticmethod
    def filter_pc_only(pc, label):
        median = np.median(pc, 0)
        valid_idx = PCloudsFilter.nobkg_3dmask(median, pc, label)
        if np.sum(valid_idx) < 10:
            print("{} filter out completely".format(label))
            return None
        pc = pc[valid_idx, :]
        return pc