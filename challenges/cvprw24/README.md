

### Data format for image based tasks
The data is split into train, val and test set. Inside each folder there is images, gt and masks subfolder, which stores GT annotation, the corresponding RGB and human object masks. Each image is idendified with a unique id, which is used to find the associated mask and annotation files.
The annotation file stores a dict of the following:
```bash 
{
    'pose': (156,) # SMPL-H body pose parameters
    'betas': (10,) # SMPL-H body shape parameters
    'trans': (3,) or (1, 3) # the SMPL-H global translation parameters
    'gender': male/female # gender of this specific subject
    'obj_angle': (3,) # axis angle representation of the object rotation, convert to 3x3 matrix: scipy.spatial.transform.Rotation.from_rotvec(a).as_matrix()
    'obj_trans': (3,) # object translation parameters
    'verts_smpl': (N_s, 3) # GT SMPL-H vertices
    'verts_obj': (N_o, 3) # GT object vertices
}
```
The object template meshes are packed in [this file](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw24/ref_hoi.zip), 
which can be indexed via `data['templates'][<obj_name>]`.

### Date format for video based HOI tracking task
The full dataset is also split into train, val and test. Each video is also identified with a unique id.
```bash
|--train
|----<video_id>.mp4 # the input RGB video
|----<video_id>_hum_mask.mp4 # the video of the human masks
|----<video_id>_obj_mask.mp4 # the video of the object masks
|----<video_id>_metadata.pkl # information regarding the subject gender and object name
|----<video_id>_gt.pkl # the video of the object masks
|--val  # save format as train
|--test # same format as test 
```
The videos can be extracted to frames simply via:
```python
import imageio

video = imageio.get_reader(video_file)
for i, frame in enumerate(video):
    imageio.imsave(outfile, frame)
```
Inside the GT pickle file, we provide the GT annotation as following:
```bash
{
  'smplh_poses': (T, 156) # SMPLH pose parameters, in total T frames in this video
  'smplh_betas': (T, 10)  # SMPLH shape parameters
  'smplh_trans': (T, 3)   # SMPLH translation parameters
  'obj_rot': (T, 3, 3)    # object rotation matrices
  'obj_trans': (T, 3)     # object translation parameters
}
```
Same as the image based tasks, the object template meshes are packed in [this file](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw24/ref_hoi.zip).