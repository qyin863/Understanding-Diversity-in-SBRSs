# Understanding-Diversity-in-SBRSs

The official Pytorch code for paper 'Understanding Diversity in Session-Based Recommendation'.

It concludes the pytorch implementation codes for SOTA baselines in session-based recommendation used in our paper (as in folder '/seren/model/').

Second, the folder '/seren/utils/' contails the code for dataset pre-processing, Dataset Class definition, dataset processing for model input,  evaluation metrics' defination, and etc. 

Third, the folder '/seren/tune_log/hypers' contains the optimal hyper-parameter setting for each SOTA SBRS on selected datasets. The dataset information is listed in '/seren/tune_log/readme.txt'.

Run besttest.py file to train and test the model.
```bash
python python besttest.py --model=narm --dataset=tmall
```

