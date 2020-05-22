#import relevant libaries
import csv
import math
import cv2
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU, Reshape
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.vis_utils import plot_model

######################################################
def generator(samples, batch_size):
    num_samples = len(samples)
    correction = 0.2  # correction angle used for the left and right images

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angle = []

            for line in batch_samples:
                ##Center##
                #image
               
                path=line[0]
                tokens=path.split('/')
                filename=tokens[-1]
                local_path="./IMG/" + filename
                image=cv2.imread(local_path)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                images.append(image)
                imagelr=cv2.flip(image,1)
                images.append(imagelr)
                imageinv = cv2.bitwise_not(image)
                images.append(imageinv)
                imageinvlr = cv2.bitwise_not(imagelr)
                images.append(imageinvlr)
                """
                imagebr=tf.image.random_brightness(image, 0.2, seed=None)
                images.append(imagebr)
                imagebrlr=tf.image.random_brightness(imagelr, 0.2, seed=None)
                images.append(imagebrlr)
                imageco=tf.image.random_contrast(image, 0.2, 0.5)
                images.append(imageco)
                imagecolr=tf.image.random_contrast(imagelr, 0.2, 0.5)
                images.append(imagecolr)
                """
                #angle
                steering=float(line[3])
                steeringlr=steering * -1.0
                angle.append(steering)
                angle.append(steeringlr)
                angle.append(steering)
                angle.append(steeringlr)
                """
                angle.append(steering)
                angle.append(steeringlr)
                angle.append(steering)
                angle.append(steeringlr)
                """
                ##left##
                #image
                path=line[1]
                tokens=path.split('/')
                filename=tokens[-1]
                local_path="./IMG/" + filename
                image=cv2.imread(local_path)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                images.append(image)
                imagelr=cv2.flip(image,1)
                images.append(imagelr)
                #angel
                angle.append(steering+correction)
                angle.append(steeringlr+correction)
                ##right##
                #image
                path=line[2]
                tokens=path.split('/')
                filename=tokens[-1]
                local_path="./IMG/" + filename
                #print(local_path)
                image=cv2.imread(local_path)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                images.append(image)
                imagelr=cv2.flip(image,1)
                images.append(imagelr)
                #angel
                angle.append(steering-correction)
                angle.append(steeringlr-correction)
              
                                  
                    
            X_train = np.array(images)
            Y_train = np.array(angle)
            
            #print(X_train.shape)
            #print(Y_train.shape)
            
            yield shuffle(X_train, Y_train)
            
            
batch_size=8
###########################################################
#create empty arrays for each line of the csv file, images and angle
lines=[]
#images=[]
#angle=[]
#read images from csv file and append each line to the lines list
with open('./driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.25)
"""        
#correction value that is added/subtracted to/from the angle i the image is from the left or right camera
correction=0.2

#i is initialized with 0 (is used to count each line from the csv file)
i=0

#loop through each line of the lines list
for line in lines:
    ##Center##
    #image
    path=line[0]
    tokens=path.split('/')
    filename=tokens[-1]
    #filename for the center image
    local_path="./IMG/" + filename
    #read image
    image=cv2.imread(local_path)
    #convert image to hsv color space
    imagehsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #convert image to hls color space
    imagehls=cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #invert image
    imageinv = cv2.bitwise_not(image)
    append images
    images.append(imageinv)
    images.append(image)
    images.append(imagehsv)
    images.append(imagehls)
    #flip iage and append
    imagelr=cv2.flip(image,1)
    images.append(imagelr)
    #angle
    steering=float(line[3])
    #multiply with -1 for flipped ige
    steeringlr=steering * -1.0
    #append steering angle multiple times
    angle.append(steering)
    angle.append(steering)
    angle.append(steering)
    angle.append(steering)
    angle.append(steeringlr)
    
    ##left##
    #image
    path=line[1]
    tokens=path.split('/')
    filename=tokens[-1]
    local_path="./IMG/" + filename
    image=cv2.imread(local_path)
    #imagehsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images.append(image)
    #images.append(imagehsv)
    #imagelr=cv2.flip(image,1)
    #images.append(imagelr)
    #angel
    #angle.append(steering+correction)
    angle.append(steering+correction)
    #angle.append(steeringlr+correction)
    
    ##right##
    #image
    path=line[2]
    tokens=path.split('/')
    filename=tokens[-1]
    local_path="./IMG/" + filename
    #print(local_path)
    image=cv2.imread(local_path)
    #imagehsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images.append(image)
    #images.append(imagehsv)
    #imagelr=cv2.flip(image,1)
    #images.append(imagelr)
    #angel
    angle.append(steering-correction)
    #angle.append(steering-correction)
    #angle.append(steeringlr-correction)
    
    print(i)
    i+=1
#print('a: ',angle)

#Memory Error here:
#Convert training data to numpy arrays    
X_train=np.array(images)
Y_train=np.array(angle)
#Architecture (LeNet5)
print(X_train.shape)
print(Y_train.shape)

#Shuffle training data
X_train, Y_train = shuffle(X_train,Y_train)
##############################################

model=Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,(3,3),activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,(3,3),activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,epochs=6)
model.summary()
"""
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


#Architecture
model=Sequential()
#lambda layer to normalize images
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
#crop upper and lower part of images
model.add(Cropping2D(cropping=((65,20),(0,0))))

#model.add(Reshape((120,200,3), input_shape=(75,320,3)))


#add convolutional and pooling layer
model.add(Convolution2D(6,(5,5)))#,activation='elu'))
model.add(ELU())
model.add(MaxPooling2D())
model.add(Convolution2D(16,(5,5)))#,activation='relu'))
model.add(ELU())
model.add(MaxPooling2D())
model.add(Convolution2D(28,(3,3)))#,activation='relu'))
model.add(ELU())
model.add(MaxPooling2D())
#flatten and add dense layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(16))  
model.add(Dense(1))
#compile using adam optimizer
model.compile(optimizer=Adam(lr = 0.001),loss='mse')
#validation split of 20 percent and number of epochs to 8
#model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,epochs=2)
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), epochs=10, validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.summary()

#save model
model.save('model.h5')

# python drive.py model.h5 run1



