# 3rd RHOBIN Challenges
  [Workshop website](https://rhobin-challenge.github.io/) | [Human recon](https://codalab.lisn.upsaclay.fr/competitions/21687) | [Object 6DoF](https://codalab.lisn.upsaclay.fr/competitions/21755) | [Joint recon (template)](https://codalab.lisn.upsaclay.fr/competitions/21752) | [Joint recon (template-free)](https://codalab.lisn.upsaclay.fr/competitions/21680) | [Joint tracking](https://codalab.lisn.upsaclay.fr/competitions/21697) | [Contact estimation TBD](TBD)


This folder provides the evaluation code for the 2nd Rhobin Challenges held in conjunction with the [CVPR'25 workshop](https://rhobin-challenge.github.io/). 
- Overview

- About the data 

- Evaluation

- Submission 

- Citations 

## Overview
Following the success of the previous RHOBIN challenges at CVPR'23 and CVPR'24, we are hosting the third edition of the challenge: reconstruction of human-object interaction from monocular RGB cameras. 

We have seen promising progress in reconstructing human body mesh or estimating 6DoF object pose from single images. However, most of these works focus on occlusion-free images which are not realistic for settings during close human-object interaction since humans and objects occlude each other. This makes inference more difficult and poses challenges to existing state-of-the-art methods.
Similarly, methods estimating 3D contacts have also seen rapid progress, but are restricted to scanned or synthetic datasets, and struggle with generalization to in-the-wild scenarios.  In this workshop, we want to examine how well the existing human and object reconstruction and contact estimation methods work under more realistic settings and more importantly, understand how they can benefit each other for accurate interaction reasoning. The recently released BEHAVE (CVPR'22), InterCap (GCPR?22) and DAMON (ICCV?23) datasets enable joint reasoning about human-object interactions in real settings and evaluating contact prediction in the wild. Based on these datasets, this competition is split into five tracks:


- 3D human reconstruction from monocular RGB images | [website](https://codalab.lisn.upsaclay.fr/competitions/21687).
- 6DoF object pose estimation from monocular RGB images | [website](https://codalab.lisn.upsaclay.fr/competitions/21755).
- Joint human and object reconstruction from monocular RGB images (**template based**) | [website](https://codalab.lisn.upsaclay.fr/competitions/21752).
- Joint human and object reconstruction from monocular RGB images (**template free**) | [website](https://codalab.lisn.upsaclay.fr/competitions/21680).
- Joint tracking of human and object from monocular RGB video | [website](https://codalab.lisn.upsaclay.fr/competitions/21697). 
- Estimating contacts from single RGB image | [TBD]. 

The winner of each track will be invited to give a talk in our CVPR'25 Rhobin workshop.

## Updates 
- Feb 16: websites up, hello world!  

## About the data
We use BEHAVE, InterCap and IMHD for the first five tasks. For convenience, we process and pack the files into tar files. For download links, please refer to their challenge webpages.  
These files are packed in the following format: 

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
The object template meshes are packed in [this file](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw25/ref_hoi.zip), 
which can be indexed via `data['templates'][<obj_name>]`.

### Data format for video based HOI tracking task
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

## Evaluation
Please refer to each competition webpages for detailed definition of the evaluation metrics. 

We provide reference evaluation code which is used by codalab server to compute the results in the leaderboard. To run the code, you should have the files organized as follows:
```bash
|ROOT
|--ref
|----ref.pkl           # reference GT file
|----SMPLH_female.pkl  # SMPLH model file, we use the MANO v1.2 version. 
|----SMPLH_male.pkl
|----SMPLH_neutral.pkl # the neutral SMPLH model used for the amass project, see cvprw25/convert_smplh_neutral.py
|--res
|----results.pkl       # your prediction results 
```
Participants should pack their results into one pkl file named as `results.pkl`, and zip it as a zipfile which can be uploaded to the codalab portal. 
The data format of the file `results.pkl` should be:
```bash
{
    <image_id> or <video_id>: # a unique id to identify the image or video sequence  
    {
        # Human results, required for human reconstruction and joint reconstruction task. 
        pose: np array (156,) of SMPL pose parameters, or (T, 156) for video-based task, where T is the number of frames
        betas: np array (10,) of SMPL shape parameters, or (T, 10) for video-based task
        trans: np array  (3,) of SMPL global translation parameters, or (T, 3) for video-based task
        joints: [optional] np.float16 array (24, 4) of SMPL body joints, alternative format for human recon/joint HOI recon.

        # Object results, required for object 6DoF and joint reconstruction track
        obj_rot: np array (3x3) of object rotation parameters, or (T, 3, 3) for video-based task
        obj_trans: np array (3,) of the object translation parameters, or (T, 3) for video-based task
        
        # Human + object for template-free reconstruction
        hum_pts: np array (6000x3), reconstructed human represented as points
        obj_pts: np array (6000x3), reconstructed object represented as points
    },
    ...
} 
```
You can then run the evaluation with:
```bash
python cvprw24/evaluate_<task>.py ROOT <output dir>
```
where `<task>` can be `human`, `object`, `joint`, `tracking`, or `joint_tfree`, depending on your specific task. 

**Important note:** We do not support uploading human results as vertices anymore because the total file size exceeds 300MB. 
If your method does predict only SMPL vertices, pleae refere to [SMPLFitter](https://github.com/isarandi/smplfitter) to convert your results as SMPLH parameters. 
It is easy to setup and can run at more than 1k frames per seconds.

## Submission
Each participant is allowed to submit maximum 10 times per day and 100 submissions in total. 

Participants must pack their results into one pkl file named as results.pkl and submit it as zip file. The pkl data should be organized as described above. 

The submission portal can be found in each individual competition website. 

### Deadline
The competition server will be closed at May 30 23:59, 2025 UTC. NO submissions will be allowed after the deadline. 

## Citations
If you use the code or the dataset, please considerc cite:
```bib
@inproceedings{bhatnagar22behave,
  title={Behave: Dataset and method for tracking human object interactions},
  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya A and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15935--15946},
  year={2022}
}

@inproceedings{huang2022intercap,
    title        = {{InterCap}: {J}oint Markerless {3D} Tracking of Humans and Objects in Interaction},
    author       = {Huang, Yinghao and Taheri, Omid and Black, Michael J. and Tzionas, Dimitrios},
    booktitle    = {{German Conference on Pattern Recognition (GCPR)}},
    volume       = {13485},
    pages        = {281--299},
    year         = {2022}, 
    organization = {Springer},
    series       = {Lecture Notes in Computer Science}
}

@InProceedings{zhao2024imhoi,
    author    = {Zhao, Chengfeng and Zhang, Juze and Du, Jiashen and Shan, Ziwei and Wang, Junye and Yu, Jingyi and Wang, Jingya and Xu, Lan},
    title     = {I'M HOI: Inertia-aware Monocular Capture of 3D Human-Object Interactions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {729-741}
}

@inproceedings{xie2022chore,
    title = {CHORE: Contact, Human and Object REconstruction from a single RGB image},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {October},
    organization = {{Springer}},
    year = {2022},
}

@inproceedings{xie2023vistracker,
    title = {Visibility Aware Human-Object Interaction Tracking from Single RGB Camera},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2023},
}

@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}
```
