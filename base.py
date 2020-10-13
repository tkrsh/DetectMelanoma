import pandas as pd 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os 
import operator
from scipy import ndimage, misc
import cv2
from sklearn.utils import shuffle 
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/home/tkrsh/tflogs/", profile_batch=5,write_graph=False) // Setup TensorBoard For Monitoring Training

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

################################## PARAMETERS ##############################################################


IMG_SHAPE=(216,216)
input_shape=(216,216,3)

tf.keras.backend.clear_session()

################################### DIRECTORIES AND DATAFRAME ############################################### 

directory_main="/media/tkrsh/ebbc93a5-618a-433c-b301-2406c8ffeca7/siim-isic-melanoma-classification/submission.csv"
dirctory_train="/media/tkrsh/ebbc93a5-618a-433c-b301-2406c8ffeca7/siim-isic-melanoma-classification/jpeg/train"
directory_test="/media/tkrsh/ebbc93a5-618a-433c-b301-2406c8ffeca7/siim-isic-melanoma-classification/jpeg/test"

df= pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
sub=pd.read_csv("sample_submission.csv")

df['image_name']=df['image_name'] + '.jpg'
test['image_name']=test['image_name'] + '.jpg'

##################################### TABULAR PREPROCESSING ########################################################

df=shuffle(df)
train      = df

##################################### IMAGE CROPING AND RESIZING ################################################### 

def cv2_clipped_zoom(img, zoom_factor):
  
    height, width = img.shape[:2] 
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def crop_and_zoom(img):
    bounding=(216,216)
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return cv2_clipped_zoom(img[slices],2)



########################################################################################################################## 

train_image_generator=ImageDataGenerator(rescale=1./255,preprocessing_function=crop_and_zoom)
train_data_gen=train_image_generator.flow_from_dataframe(train,directory=dirctory_train,x_col='image_name',y_col='benign_malignant',class_mode='binary',batch_size=8,target_size=IMG_SHAPE)


validation_gen=ImageDataGenerator(rescale=1./255)
 validation_data_gen=validation_gen.flow_from_dataframe(validation,directory=dirctory_train,x_col='image_name',y_col='benign_malignant',class_mode='binary',batch_size=32,target_size=IMG_SHAPE)


test_generator=ImageDataGenerator(rescale=1./255,preprocessing_function=crop_and_zoom)
test_data_gen=test_generator.flow_from_dataframe(dataframe=test,directory=directory_test,x_col="image_name",class_mode=None,target_size=IMG_SHAPE,shuffle=False)

####################################           L R SCHEDULER           ############################################# 

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

#################################### DEFINING MODEL: NEURAL NETWORK ###################################################### 


base_model=tf.keras.applications.EfficientNetB0(input_shape=input_shape,include_top=False, weights='imagenet')

base_model.trainable = True

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')

model = tf.keras.Sequential([base_model,global_average_layer,
  prediction_layer
])

model.compile(optimizer=tf.optimizers.Adam(lr=0.0001),loss='BinaryCrossentropy',metrics=[tf.keras.metrics.AUC()])

history=model.fit(train_data_gen,epochs=3,callbacks=[lr_scheduler])

########################################################################################################################### 



sub['target']=model.predict(test_data_gen) 

sub.to_csv(directory_main,index=False)          
