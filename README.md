# ViTST

This is an official implementation of the paper: "Time Series as Images: Vision Transformer for Irregularly Sampled Time Series." 

## Overview
We consider the irregularly sampled multivariate time series modeling from a whole new perspective: transforming irregularly sampled time series into line graph images and adapting powerful vision transformers to perform time series classification in the same way as image classification.
With a few lines of code to transform the time series into line graph images, any vision model can be used to handle any type of time series.

<!-- ![Raindrop idea] -->
<!-- (images/fig1.png "Idea of Raindrop.") -->
<p align="center">
    <img src="pics/illustration.png" width="720" align="center">
</p>


## Getting Started

We conduct experiments on three irregular time series datasets P19, P12, and PAM, and several regular time series datasets from [UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com/index.php).

### Irregular Time Series Datasets
We use the data processed by [Raindrop](https://github.com/mims-harvard/Raindrop). 

The raw data can be found at:

**(1)** P19: https://physionet.org/content/challenge-2019/1.0.0/

**(2)** P12: https://physionet.org/content/challenge-2012/1.0.0/

**(3)** PAM: http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

The processed datasets can be obtained at:

**(1)** P19 (PhysioNet Sepsis Early Prediction Challenge 2019) https://doi.org/10.6084/m9.figshare.19514338.v1

**(2)** P12 (PhysioNet Mortality Prediction Challenge 2012) https://doi.org/10.6084/m9.figshare.19514341.v1

**(3)** PAM (PAMAP2 Physical Activity Monitoring) https://doi.org/10.6084/m9.figshare.19514347.v1


Follow these two steps to create the dataset:
1. Get the processed data, unzip them, and put the files in ```dataset``` folder.
2. Run the following commands in turn to create the images: ```cd dataset/P12data/```, ```python ParamDescription.py```, ```python ConstructImage.py```

### Regular Time Series Datasets
You can download the datasets at http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip. 

Create a folder ```Classification``` in the ```TSRAdara``` folder. Run ```PlotMarkers.py``` and ```ConstructDataset.py``` to create the images.


### Training
For the dataset containing static features, such as P19 and P12, go to the ```code/Vision-Text/``` folder and run the script ```vtcls_script.sh``` to start training:
```
cd code/Vision-Text/
sh vtcls_script.sh
```

For the other datasets, go to the ```code/Vision/``` folder and run the script ```imgcls_script.sh``` to start training: 
```
cd code/Vision/
sh imgcls_script.sh
```

### Self-supervised learning
If you want to self-supervised learning pretrain the model, go to the ```code/Vision/``` folder and run the script ```imgmim_script.sh``` to start training: 
```
cd code/Vision/
sh imgmim_script.sh
```

### Run baseline methods
We use the code provided by [Raindrop](https://github.com/mims-harvard/Raindrop). The code for the following baseline methods are placed in ```dataset/raindrop``` folder: Transformer, Trans-mean, GRU-D, SeFT and mTAND. See details of these baselines in our paper. 

Starting from root directory ```dataset/raindrop```, you can run models as follows:

- Raindrop
```
python Raindrop.py
```

- Transformer
```
cd baselines
python Transformer_baseline.py
```

- Trans-mean
```
cd baselines
python Transformer_baseline.py --imputation mean
```

- GRU-D
```
cd baselines
python GRU-D_baseline.py
```

- SeFT
```
cd baselines
python SEFT_baseline.py
```

- mTAND
```
cd baselines/mTAND
python mTAND_baseline.py
```

- IP-Net
```
cd baselines/IP_Net/src
python IP_Net_baseline.py
```

- MTGNN
```
cd baselines
python MTGNN_baseline.py
```

- DGM2-O
```
cd baselines
python DGM2_baseline.py
```

<!-- ## Acknowledgement

We appreciate Huggingface and the following github repo very much for the valuable code base and datasets:

https://github.com/mims-harvard/Raindrop

https://github.com/gzerveas/mvts_transformer
 -->
