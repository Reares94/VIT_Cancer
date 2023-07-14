# Vit_Cancer
Multiclass classification for histological cancer
-----------------------------------------------------------------------------------------------

This project implements the Vision Transformer architecture applied to PatchcameLyon using the Tensorflow library to solve the multiclass classification problem for histological cancer.The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue otherwise not.The dataset was divided into 4 tumor subfolders to apply the multiclass classification.

------------------------------------------------------------------------------------------------

How can I produce the results? 

1 You need to download the dataset from this link https://www.kaggle.com/competitions/histopathologic-cancer-detection/data. The dataset has been manipulated to achieve a multiclass problem.

2 Download the code I wrote and try to understand how it was organized by helping you with the comments

3 The dataset was divided in the proportion 80: 10 :10 with respect to training, validation and testing

4 The network is evaluated with respect to the accuracy and loss in the different subdivisions of the dataset

5 You can try changing the parameters in the configuration.py file to get other performance

6.You can run the train.py file to train the network and also calculate the specifications related to the validation part. Only after the weights have been automatically saved in the model.h file you can run the test.py

--------------------------------------------------------------------------------------------------

How do I install it?

1 This project was implemented on macOS by enabling the use of the M1 GPU chip and using the tensorflow library. Search the net how to enable it


--------------------------------------------------------------------------------------------------
Architecture and implemented equations, tumor dataset, patch_embedding

<img width="772" alt="vision_transformer" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/fcaf5cb5-fc83-4ba3-b765-6dc91a654c3f">
<img width="1000" alt="Equations" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/2d232279-d4b4-4207-ad16-d31160f22591">

![Griglia_tumori](https://github.com/Reares94/VIT_Cancer/assets/93512390/bb2abef8-408c-40a0-a906-c77555ca300f)
<img width="465" alt="patch_cancer" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/5ed0fe8e-24c7-4dcb-8827-0d6dd34058c6">



--------------------------------------------------------------------------------------------------

References

[1]https://arxiv.org/pdf/2010.11929.pdf

[2]https://www.youtube.com/watch?v=Ssndsjh1Zqk

[3]https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

[4]https://arxiv.org/pdf/1803.08494.pdf

[5]https://arxiv.org/pdf/1606.08415.pdf

[6]https://arxiv.org/pdf/1412.6980.pdf


