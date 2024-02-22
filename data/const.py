"""
some meta data for the dataset
"""


_sub_gender = {
"Sub01": 'male',
"Sub02": 'male',
"Sub03": 'male',
"Sub04": 'male',
"Sub05": 'male',
"Sub06": 'female',
"Sub07": 'female',
"Sub08": 'female',
}

OBJ_NAMES=['backpack', 'basketball', 'boxlarge', 'boxlong', 'boxmedium',
           'boxsmall', 'boxtiny', 'chairblack', 'chairwood', 'keyboard',
           'monitor', 'plasticcontainer', 'stool', 'suitcase', 'tablesmall',
           'tablesquare', 'toolbox', 'trashbin', 'yogaball', 'yogamat']

USE_PSBODY = True # if True, use psbody library to process all meshes, otherwise use trimesh