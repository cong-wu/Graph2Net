# Graph2Net
The Official implementation for [Graph2Net: Perceptually-Enriched Graph Learning for Skeleton-Based Action Recognition](https://ieeexplore.ieee.org/document/9446181) (TCSVT 2021).

Also, on the basis of this method, we won the first place in [Multi-Modal Video Reasoning and Analyzing Competition (MMVRAC, Track 2 Skeleton-based Action Recognition)](https://sutdcv.github.io/multi-modal-video-reasoning/#/) from ICCV Workshop.


- [Prerequisite](#Prerequisite)
- [Data](#Data)
- [Training&Testing](#Training&Testing)
- [Ensemble](#Ensemble)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)
- [Contact](#Contact)


<a name="Prerequisite"></a>

# Prerequisite

- Python 3.7
- Pytorch 1.5
- Other Python libraries can be installed with `pip install -r requirements.txt`.

 
<a name="Data"></a>

# Data

## Generate the Joint data 

**Ntu-RGB+D 60 & 120**

- Download the raw data of [NTU-RGB+D](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). Put NTU-RGB+D 60 data under the directory `./data/nturgbd_raw`. Put NTU-RGB+D-120 data under the directory `./data/nturgbd120_raw`.
- For NTU-RGB+D 60, preprocess data with `python data_gen/ntu_gendata.py`. 
- For NTU-RGBD+120, preprocess data with `python data_gen/ntu120_gendata.py`. 

**Kinetics-400 Skeleton**

- Download the raw data of [Kinetics-400 Skeleton](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md). Put Kinetics-400 Skeleton data under the directory `./data/kinetics_raw/`.
- Preprocess data with `python data_gen/kinetics_gendata.py`.

**Northwestern-UCLA**

The preprocess of Northwestern-UCLA dataset is borrow from [kchengiva/Shift-GCN](https://github.com/kchengiva/Shift-GCN/issues/13#issuecomment-718395974).

- Download the raw data of [Northwestern-UCLA](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0). Put Northwestern-UCLA data under the directory `./data/nw_ucla_raw/`.

## Generate the Bone data 
- Generate the bone data with `python data_gen/gen_bone_data.py`.


<a name="Training&Testing"></a>

# Training&Testing

## Training 

We provided several examples to train Graph2Net with this repo:

- To train on NTU-RGB+D 60 under Cross-View evaluation, you can run


    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`
    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

- To train on Mini-Kinetics-Skeleton, you can run


    `python main.py --config ./config/kinetics-skeleton/train_joint.yaml`
    `python main.py --config ./config/kinetics-skeleton/train_bone.yaml`

- To train on Northwestern-UCLA, you can run


    `python main_nw_ucla.py --config ./config/northwestern-ucla/train_joint.yaml`
    `python main_nw_ucla.py --config ./config/northwestern-ucla/train_bone.yaml`


## Testing 

We also provided several examples to test Graph2Net with this repo:

- To train on NTU-RGB+D 60 under Cross-View evaluation, you can run


    `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`
    `python main.py --config ./config/nturgbd-cross-view/test_bone.yaml`

- To train on Mini-Kinetics-Skeleton, you can run


    `python main.py --config ./config/kinetics-skeleton/test_joint.yaml`
    `python main.py --config ./config/kinetics-skeleton/test_bone.yaml`

- To train on Northwestern-UCLA, you can run


    `python main_nw_ucla.py --config ./config/northwestern-ucla/test_joint.yaml`
    `python main_nw_ucla.py --config ./config/northwestern-ucla/test_bone.yaml`

The corresponding result of the above command is as follows,

|        | NTU-RGB+D 60 (Cross-View)   | Mini-Kinetics-Skeleton | Northwestern-UCLA |
| ------- | :---------: | :---------: | :---------: | 
|Joint     | 95.2                     | 42.3              | 94.4              |
|Bone     |  94.6                          | 42.1              | 92.5             |

In the `save_models` folder, we also provide the trained model parameters.

Please refer to the `config` folder for other training and testing commands. You can also freely change the train or test config file according to your needs. 

<a name="Ensemble"></a>

# Ensemble

To ensemble the results of joints and bones, run the test command we provided to generate the scores of the softmax layer.
Then combine the generated scores with:

- NTU-RGB+D 60 

    `python ensemble.py --datasets ntu/xview`

- Mini-Kinetics-Skeleton

    `python ensemble.py --datasets kinetics_min_skeleton`

- Northwestern-UCLA

    `python ensemble_nw_ucla.py`

The corresponding result of the above command is as follows,

|        | NTU-RGB+D 60 (Cross-View)   | Mini-Kinetics-Skeleton | Northwestern-UCLA |
| ------------- | :---------: | :---------: | :---------: | 
|Ensemble | 96.0 | 44.9| 95.3|


<a name="Citation"></a>

# Citation
If you find this model useful for your research, please use the following BibTeX entry.

	@ARTICLE{9446181,
	  author={Wu, Cong and Wu, Xiao-Jun and Kittler, Josef},
      journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
      title={Graph2Net: Perceptually-Enriched Graph Learning for Skeleton-Based Action Recognition}, 
      year={2021},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TCSVT.2021.3085959}
    }


<a name="Acknowledgement"></a>

# Acknowledgement
Thanks for the framework provided by [2s-AGCN](https://github.com/lshiwjx/2s-AGCN), which is source code of the published work [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html) in CVPR 2019. 


<a name="Contact"></a>

# Contact
For any questions, feel free to contact: `congwu@stu.jiangnan.edu.cn`.