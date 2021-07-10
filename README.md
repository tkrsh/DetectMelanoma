## Detecting Skin Cancer Melanoma (malignant melanoma) 

Melanoma occurs when the pigment-producing cells that give colour to the skin become cancerous.
Symptoms might include a new, unusual growth or a change in an existing mole. Melanomas can occur anywhere on the body.
Treatment may involve surgery, radiation, medication or in some cases, chemotherapy.

## Dataset  

### Society for Imaging Informatics in Medicine  

[SIIM-ISIC Melanoma Classification](https://siim.org/page/siim_isic_melanoma_classification)

The images were provided in DICOM format JPEG and TFRecord format.

[Download Dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data)

## Hardware Used For Training:

  1) TPUs on Kaggle 
  2) Local Machine  Nvidia GPU  GTX 1660 Ti 
  3) Inference done on CPU
  
## Library and Language Used :   

Python 3.6 
  1) [TensorFlow I/0](https://www.tensorflow.org/io)    // DICOM handling 
  2) [TensorFlow 2.3](https://www.tensorflow.org/)    // Deep Learning Model Implementation
  3) [Pydicom 2.0](https://www.tensorflow.org/io)     // DICOM handling
  4) [OpenCv 3.2](https://www.tensorflow.org/io)      // Image Preprocessing
  5) [Pandas 1.13](https://www.tensorflow.org/io)     // Csv Handling
  

## Conv Models Used  :

  1) [EfficientNetb0-b7](https://arxiv.org/abs/1905.11946)
  2) [DenseNet 169](https://arxiv.org/abs/1608.06993)

## Preprocessing Used (Image):
  
  1) Resizing , zoom and croping 
  2) Image Augmentation: 
  
    1) Rotation_range = 180
    2) shear_range = 0.4
    3) Horizontal And Vertical Flipping 
    4) Rescale 
    
 ## Other Techniques Used:
 
 Generated patient metadadata from DICOM  
 
 Performed 2D/3D layering utilizing different windows and slicing of DICOM Image
  
## Result Obtained :

  ROC = 0.93  
  
  Submissions were evaluated on area under the ROC curve between the predicted probability and the observed target.


  
