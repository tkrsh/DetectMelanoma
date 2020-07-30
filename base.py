import pandas as pd 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os 
base_dir=("/media/tkrsh/ebbc93a5-618a-433c-b301-2406c8ffeca7/siim-isic-melanoma-classification")

dirctory_train=base_dir + '/jpeg/train'
directory_test= base_dir + '/jpeg/test'

train= pd.read_csv(base_dir+"/train.csv")
test=pd.read_csv(base_dir+"/test.csv")
sub=pd.read_csv(base_dir+"/sample_submission.csv")
train=train[:200]
test=test[:200]
sub=train[:200]
train['image_name']=train['image_name'] + '.jpg'
test['image_name']=test['image_name'] + '.jpg'


os.listdir(dirctory_train)

# train_image_generator=ImageDataGenerator(rescale=1./255,validation_split=0.2)
# train_data_gen=train_image_generator.flow_from_dataframe(train,directory=dirctory_train,x_col='image_name',y_col='benign_malignant',class_mode='binary',batch_size=32)

# IMG_SHAPE=(32,32,3)

# #base_model=tf.keras.applications.ResNet152(input_shape=IMG_SHAPE,include_top=False, weights='imagenet',classes=1)

# #base_model.trainable = False


# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(1)

# model = tf.keras.Sequential([
#   global_average_layer,
#   prediction_layer
# ])

# test_generator=ImageDataGenerator(rescale=1./255.)
# test_data_gen=test_generator.flow_from_dataframe(dataframe=test,directory=directory_test,x_col="image_name",y_col=None,class_mode=None)


# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.losses.BinaryCrossentropy(from_logits=True),
#               metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])


# history=model.fit(train_data_gen,epochs=2)
# sub=sub.drop("target",axis=1)

# sub['target']=(np.argmax(model.predict(test_data_gen,use_multiprocessing=True), axis=-1))
# sub.to_csv("submission.csv")