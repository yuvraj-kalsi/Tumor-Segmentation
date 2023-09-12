
# 🚀 Tumor Segmentation of Breast & Thyroid Ultrasound Images

## Problem Statement

In the field of medical imaging, the manual identification and diagnosis of tumors in ultrasound scans can be highly inefficient and error-prone. Doctors and radiologists are tasked with the challenging job of scrutinizing numerous images to detect tumors, which can vary in size, location, and appearance. This manual process is time-consuming and can lead to inconsistencies and misdiagnoses due to human error, fatigue, and variations in expertise.

## Proposed Solution
The implementation of deep learning models can solve this problem efficiently such that:

* It can significantly enhance the accuracy and efficiency of tumor detection

* It can eliminate the need for extensive manual inspection by doctors and radiologists.

* Automated tumor detection can significantly accelerate the diagnostic process.

* This will ensure that every patient's scan is analyzed using the same criteria, leading to more reliable diagnoses.

## What is Semantic Segmentation ?  

This is technique which involves assigning a specific class label to each pixel in an image, effectively dividing the image into meaningful regions or segments based on the objects or structures present. It plays a crucial role by precisely delineating the tumor boundaries within the image. By assigning each pixel in the image as either part of the tumor or background, this method allows for the accurate identification and localization of tumors, which is essential for diagnosis, treatment planning, and monitoring disease progression. 

## Implementation

**Dataset used in the program:** 🔗[Breast Ultrasound](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), [Thyroid Ultrasound](https://www.kaggle.com/datasets/eiraoi/thyroidultrasound)

The dataset contains:
```bash
First directory of USG Images
Second directory of corresponding Masked Images
```

**Libraries used in the program:**
* [NumPy](https://numpy.org/doc/stable/) and [Pandas](https://pandas.pydata.org/docs/) - Data Manipulation
* [Matplotlib](https://matplotlib.org/stable/index.html) - Data Visualization
* [Tensorflow](https://www.tensorflow.org/api_docs) and [Keras](https://www.tensorflow.org/guide/keras) - Deep Learning Model Implementation
* [Sklearn](https://scikit-learn.org/0.21/documentation.html) - Models Evaluation

**Model used in the program:**

The **U-Net** is a popular convolutional neural network (CNN) architecture specifically designed for semantic segmentation tasks, including tumor segmentation in medical images. It is named "U-Net" due to its **U-shaped architecture**, which consists of an encoder and a decoder, allowing it to capture detailed information and generate precise segmentation masks. 

Data-Flow:

* Input Data:
The U-Net takes as input medical images, such as ultrasound (USG) or MRI scans, that contain tumors.

* Architecture:  
a) Encoder: Captures contextual information from the input image. It consists of a series of convolutional layers with pooling operations (typically max-pooling) that progressively reduce the spatial dimensions of the feature maps while increasing the number of feature channels

b) Bottleneck: After several encoding layers, there's a bottleneck layer that captures the most critical features.   

c) Decoder: The decoder's purpose is to upsample and reconstruct the segmentation mask. It consists of a series of up-convolutional (transposed convolution or upsampling) layers that gradually increase the spatial resolution of the feature maps.    

* Training:
During training, the network learns to minimize the difference between its predictions and the ground truth masks.


**Output:**

Breast Masking:

![output-breast](https://github.com/yuvraj-kalsi/Tumor-Segmentation/assets/84912620/e1a7b6eb-5e66-42cd-ad59-81cf826c1a53)


Thyroid Masking:





