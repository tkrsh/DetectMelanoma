## Detecting Skin Cancer Melanoma (malignant melanoma) 

Melanoma occurs when the pigment-producing cells that give colour to the skin become cancerous.
Symptoms might include a new, unusual growth or a change in an existing mole. Melanomas can occur anywhere on the body.
Treatment may involve surgery, radiation, medication or in some cases, chemotherapy.

Dataset- (https://www.kaggle.com/c/siim-isic-melanoma-classification/data)

The images were provided in DICOM format

## Library and Language Used :   

Python 3.6 
  1) TensorFlow I/0  // DICOM handling 
  2) TensorFlow 2.3  // Deep Learning Model Implementation
  3) Pydicom 2.0     // DICOM handling
  4) OpenCv 3.2      // Image Preprocessing
  

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

  ROC = 0.93 - 0.95 [Classes Were Imbalanced]
  
