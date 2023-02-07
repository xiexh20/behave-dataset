# BEHAVE Challenges
  [Workshop website](https://rhobin-challenge.github.io/) | [Human recon](https://codalab.lisn.upsaclay.fr/competitions/9280) | [Object 6DoF](https://codalab.lisn.upsaclay.fr/competitions/9336) | [Joint recon](https://codalab.lisn.upsaclay.fr/competitions/9337)


This folder provides the evaluation code for the Rhobin Challenges held in conjunction with the [CVPR'23 workshop](https://rhobin-challenge.github.io/). 



- Overview

- About the data 

- Submission 

- Evaluation

- Citations 


## Overview 
We have seen promising progress in reconstructing human body mesh or estimating 6DoF object pose from single images. However, most of these works focus on occlusion-free images, which is not realistic for settings during close human-object interaction since humans and objects occlude each other. This makes inference more difficult and poses challenges to existing state-of-the-art methods. In this challenge, we want to examine how well the existing human and object reconstruction methods work under more realistic settings and more importantly, understand how they can benefit each other for accurate interaction reconstruction. The recently released BEHAVE dataset (CVPR'22), enables for the first time joint reasoning about human-object interactions in real settings. Based on the BEHAVE dataset, this competition is split into three tracks:

- Human reconstruction | [website](https://codalab.lisn.upsaclay.fr/competitions/9280)
- Object 6DoF pose estimation | [website](https://codalab.lisn.upsaclay.fr/competitions/9336)
- Joint human object reconstruction | [website](https://codalab.lisn.upsaclay.fr/competitions/9337)

The winner of each track will be invited to give a talk in our CVPR'23 workshop. 

## About the data
The BEHAVE dataset is used for all three tracks. The dataset captures realistic human interacting with 20 different objects in natural environments. It comes with multi-view RGB images paired with (pseudo) ground truth SMPL and object mesh registrations. The evaluation will be based on the annotated SMPL and object meshes. 

Participants are allowed to train their methods on the BEHAVE training sequences specified by this file and any other sources except the BEHAVE test set. 

For convenience, the download links are listed below:

Training sequences: [part 1 (3.4G)](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part1.zip), [part 2 (6.9G)](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part2.zip), [part 3 (5.1G)](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part3.zip). 
Test input images: [all data](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/test_images.zip).

By downloading the data, you agree the licence described in [this website](https://virtualhumans.mpi-inf.mpg.de/behave/license.html).

## Submission 

It is NOT mandatory to submit a report for your method. However, we DO encourage you to fill in [this form](https://forms.gle/oK8JXZb6tjDe5HKg7) about the additional training data you used. 

Each participant is allowed to submit maximum 5 times per day and 100 submissions in total. 

Participants must pack their results into one pkl file named as `results.pkl` and submit it as a single `zip` file. The pkl data should be organized as follows:
```python
{
    seq_name: {
        # metadata
        frames: list of input image frame times,
        gender: male/female,

        # results of human reconstruction, required for human and joint reconstruction track
        poses: np array (Tx72) of SMPL pose parameters,
        betas: np array (Tx10) of SMPL shape parameters,
        trans: np array  (Tx3) of SMPL global translation parameters,

        # results of object reconstruction, required for object 6DoF and joint reconstruction track
        obj_rots: np array (Tx3x3) of object rotation parameters,
        obj_trans: np array (Tx3) of the object translation parameters,
        obj_scales: optional, a list of T object scale parameters, if not given, it will be set to one.
    },
    ...
}
```

Example submissions for different tracks can be found below:
- [Human](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/examples/mocap_results.pkl)
- [Object 6DoF pose estimation](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/examples/chore_object.pkl)
- [Joint human object reconstruction](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/examples/chore_joint.pkl)

## Evaluation
The evaluation metrics can be found on the website of each tracks respectively. 

You can also run the evaluation code provided here. Please follow the steps below. 

### Setup the environment
You can install the required python packages for the evaluation by `pip install -r requirements.txt`.

### Run evaluation code
To run the code, you need to pack your results as described above and organize the GT files as below:
```bash 
DATA_ROOT
--ref
----ref.pkl
--res
----results.pkl 
```
where `ref.pkl` is the ground truth data which you can download [here](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/ref.pkl).

With that data structure, you can run evaluation code as:
```bash
python evaluate_[track].py [DATA_ROOT] [OUT_DIR]
```
here `evaluate_[track].py` can be `evaluate_human.py`, `evaluate_object.py` or `evaluate_joint.py`. 
For example, to evaluation human reconstruction results, you can run: 
```bash 
python evaluate_human.py [DATA_ROOT] [OUT_DIR]
```



## Citations
If you use the code, please cite:
```bibtex
@inproceedings{bhatnagar22behave,
  title={Behave: Dataset and method for tracking human object interactions},
  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya A and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15935--15946},
  year={2022}
}
@inproceedings{xie2022chore,
    title = {CHORE: Contact, Human and Object REconstruction from a single RGB image},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {October},
    organization = {{Springer}},
    year = {2022},
}
```





