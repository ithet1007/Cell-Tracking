# Cell-Tracking
Cell tracking using deep neural networks with multi-task learning

Copyright (C) Tao He, Hua Mao, and Zhang Yi.
All rights reserved.
The code is based on NaiYan Wang, thanks to him for sharing his code
The CNN source code comes from matlab toolbox
single object cell tracking

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

## Code Running
please matlab run run_MTT.m

## dataset 
All 80 labeled cell sequences are in "samples" directoty, the label file is samples/groundtruth.mat

## Network implements
The main network setting is implemented in initMTT.m. Network training using CNN/cnntrain.m

## The positive sample queue
Implemented in pos_queue.m

## Plot test results
We public three results in "results" directory. Three plotting method are supported in get_location.m, get_precision_plot.m, and get_success_plot.m

More details please refer to our paper: url = "http://www.sciencedirect.com/science/article/pii/S0262885616302001"
