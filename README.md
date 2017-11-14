# Cell-Tracking
Cell tracking using deep neural networks with multi-task learning

Copyright (C) Tao He, Hua Mao, and Zhang Yi.
All rights reserved.
The code is based on NaiYan Wang, thanks to him for sharing his code.
The CNN source code comes from matlab toolbox.

# Abstract
Cell tracking plays crucial role in biomedical and computer vision areas. As cells generally have frequent
deformation activities and small sizes in microscope image, tracking the non-rigid and non-significant cells
is quite difficult in practice. Traditional visual tracking methods have good performances on tracking rigid
and significant visual objects, however, they are not suitable for cell tracking problem. In this paper, a novel
cell tracking method is proposed by using Convolutional Neural Networks (CNNs) as well as multi-task
learning (MTL) techniques. The CNNs learn robust cell features and MTL improves the generalization performance
of the tracking. The proposed cell tracking method consists of a particle filter motion model, a
multi-task learning observation model, and an optimized model update strategy. In the training procedure,
the cell tracking is divided into an online tracking task and an accompanying classification task using the
MTL technique. The observation model is trained by building a CNN to learn robust cell features. The tracking
procedure is started by assigning the cell position in the first frame of a microscope image sequence. Then,
the particle filter model is applied to produce a set of candidate bounding boxes in the subsequent frames.
The trained observation model provides the confidence probabilities corresponding to all of the candidates
and selects the candidate with the highest probability as the final prediction. Finally, an optimized model
update strategy is proposed to enable the multi-task observation model for the variation of the tracked cell
over the entire tracking procedure. The performance and robustness of the proposed method are analyzed
by comparing with other commonly-used methods. Experimental results demonstrate that the proposed
method has good performance to the cell tracking problem.

If you reuse our codes or our dataset, please cite our paper: url = "http://www.sciencedirect.com/science/article/pii/S0262885616302001",

@article{HE2017142,
title = "Cell tracking using deep neural networks with multi-task learning",
journal = "Image and Vision Computing",
volume = "60",
pages = "142 - 153",
year = "2017",
note = "Regularization Techniques for High-Dimensional Data Analysis",
issn = "0262-8856",
doi = "https://doi.org/10.1016/j.imavis.2016.11.010",
author = "Tao He and Hua Mao and Jixiang Guo and Zhang Yi",
}
## dataset 
Our dataset contains two part: offline training data and online training data. We public our datasets in our homepage url:
http://legacy.machineilab.org/users/hetao/, you can download from my homepage.

The offline dataset contains source dataset and processed dataset which is saved as 'offline_data.mat', you can download and used in our code. (ref to 'CNN/cellclassificationmain.m' for more details.)

The oneline dataset contains 80 cell tracking sequences. All 80 labeled cell sequences are in "samples" directoty, the label file is samples/groundtruth.mat. In our code, we just give one example, other data could be available in my homepage.

In the meantime, we still give our pre_trained cell tracking cnn model saved as 'cnn_model.mat', which could be directly used in 'run_tracker.m'.

## Code Running
offline training stage : please run CNN/cellclassificationmain.m.
online tracking stage : please run run_MTT.m. By the way, you can directly use our 'cnn_model.mat' for online tracking. 

## Network implements
The main network setting is implemented in initMTT.m. Network training using CNN/cnntrain.m

## The positive sample queue
Implemented in pos_queue.m

## Plot test results
We public three results in "results" directory. Three plotting method are supported in get_location.m, get_precision_plot.m, and get_success_plot.m

More details please refer to our paper: url = "http://www.sciencedirect.com/science/article/pii/S0262885616302001"
