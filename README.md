# Understanding-Diversity-in-SBRSs

The official Pytorch code for paper 'Understanding Diversity in Session-Based Recommendation'.

## Code
It concludes the pytorch implementation codes for SOTA baselines in session-based recommendation used in our paper (as in folder '/seren/model/').

Second, the folder '/seren/utils/' contails the code for dataset pre-processing, Dataset Class definition, dataset processing for model input,  evaluation metrics' defination, and etc. 

Third, the folder '/seren/tune_log/hypers' contains the optimal hyper-parameter setting for each SOTA SBRS on selected datasets. The dataset information is listed in '/seren/tune_log/readme.txt'.

## Usage
Run besttest.py file to train and test the model.
```bash
python python besttest.py --model=narm --dataset=tmall
```

## Citation
Please cite the following paper if you use the above content in a research paper in any way (e.g., code and evaluation metrics):
```bash
@article{yin2023diversity,
author = {Yin, Qing and Fang, Hui and Sun, Zhu and Ong, Yew-Soon},
title = {Understanding Diversity in Session-Based Recommendation},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {},
journal = {ACM Trans. Inf. Syst.}
}
```

## Acknowledgements
We refer to the following repositories to improve our code:



