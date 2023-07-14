# Vit_Cancer
Multiclass classification for histological tumors
-----------------------------------------------------------------------------------------------

This project implements the Vision Transformer architecture applied to PatchcameLyon using the Tensorflow library to solve the multiclass classification problem for histological cancer

------------------------------------------------------------------------------------------------

How can I produce the results? 

1 You need to download the dataset from this link https://www.kaggle.com/competitions/histopathologic-cancer-detection/data. The dataset has been manipulated to achieve a multiclass problem.

2 Download the code I wrote and try to understand how it was organized by helping you with the comments

3 The dataset was divided in the proportion 80: 10 :10 with respect to training, validation and testing

4 The network is evaluated with respect to the accuracy and loss in the different subdivisions of the dataset.

5 You can try changing the parameters in the configuration.py file to get other performance

6.You can run the train.py file to train the network and also calculate the specifications related to the validation part. Only after the weights have been automatically saved in the model.h file the test.py file

--------------------------------------------------------------------------------------------------

How do I install it?

1 This project was implemented on macOS by enabling the use of the M1 gpu chip and using the tensorflow library. Search the net how to enable it.


--------------------------------------------------------------------------------------------------


<img width="772" alt="vision_transformer" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/fcaf5cb5-fc83-4ba3-b765-6dc91a654c3f">

![Griglia_tumori](https://github.com/Reares94/VIT_Cancer/assets/93512390/bb2abef8-408c-40a0-a906-c77555ca300f)
<img width="465" alt="patch_cancer" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/5ed0fe8e-24c7-4dcb-8827-0d6dd34058c6">

<img width="158" alt="datasplit_total" src="https://github.com/Reares94/VIT_Cancer/assets/93512390/0baea117-c390-4ebc-b2f6-a23491da2bf2">

