
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设备
import glob
from skimage.transform import resize
import cv2
from skimage.io import imread
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Lambda
import numpy as np
np.random.seed(10)
from keras import backend as K

"""
the priori method
"""
def image_pyr_read(path):
    """
    Used to read the image pyramid.

    :param path:The path of the original image.

    :return:Output size:(3,224,224,3)
            Output pixel range:[0,255]
    """
    x = imread(path)[:,:,0:3]
    x = resize(x, (224, 224)) * 255  # cast back to 0-255 range

    img=x.copy()

    x = cv2.pyrUp(cv2.pyrDown(x))
    img_2 = x.copy()

    x = cv2.pyrUp(cv2.pyrDown(x))
    img_4 = x.copy()

    x = cv2.pyrUp(cv2.pyrDown(x))
    img_8 = x.copy()
    image_pyr=np.array([img,img_2,img_4])
    return image_pyr

def image_priori():
    """
    A priori method
    This is a method that can detect adversarial examples and estimate the strength of perturbations.
    pre0!=pre1: the parameters to detect the adversarial examples
    pre1!=pre2: the parameters to estimate the strong perturbations
    :return:
    """
    model_name = "resnet101.h5"
    model = load_model(model_name)
    path1 = "F:/my_dataset/*.png"
    count0 = 0
    count1 = 0
    count2 = 0
    for num in glob.glob(path1):
        image=image_pyr_read(num) #Read image pyramid

        y = np.argmax(model.predict(image), axis=1) #Predict image

        if y[0] != y[1]: #pre0!=pre1
            count0 += 1
        if y[1] != y[2]: #pre1!=pre2
            count1 += 1
        if y[2] != y[3]: #pre2!=pre3
            count2 += 1

        print("count0=", count0, "count1=", count1, "count2=", count2)


"""
the reconstruction model 
"""
def my_imread_pyr(path):
    """
    Used to read the image pyramid.

    :param path:The path of the original image.

    :return:Output size:(4,224,224,3)
            Output pixel range:[0,1]
    """
    x = imread(path)[:,:,0:3]
    x = resize(x, (224, 224))
    y_ori=x.copy()

    x=cv2.pyrUp(cv2.pyrDown(x))
    y2=x.copy()

    x = cv2.pyrUp(cv2.pyrDown(x))
    y4 = x.copy()

    x = cv2.pyrUp(cv2.pyrDown(x))
    y8 = x.copy()

    return y2,y4,y8,y_ori

def reconstruction_model(name):
    """
    Used to repair adversarial examples with strong perturbations.
    :param name: model name
    :return: reconstruction_model
    """
    act_name="relu"
    input_image1 = Input(shape=(224,224, 3))
    input_image2 = Input(shape=(224,224, 3))
    input_image3 = Input(shape=(224,224, 3))
    input_image=concatenate([input_image1,input_image2,input_image3],axis=-1)

    noise_patch = Lambda(lambda p: K.clip((K.random_normal(shape=K.shape(p),mean=0.0,stddev=0.09)+p),0,1))(input_image)

    y = Conv2D(16, (3, 3), padding='same', activation=act_name)(noise_patch)
    y = Conv2D(32, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(64, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(128, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(256, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(128, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(64, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(32, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(16, (3, 3), padding='same', activation=act_name)(y)
    y = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(y)

    model = Model([input_image1,input_image2,input_image3], y)
    model.compile(optimizer='adam', loss='mse')
    plot_model(
        model,
        to_file='rec_model.png',
        show_shapes=True,
        show_layer_names=False)
    model.save(name)
    return model

def model_fit(name):
    """
    Training model.
    Tensor size:[batch,224,224,3]
    :param name: Model name.
    :return:
    """
    model = load_model(name)

    batch = 100
    batch_img_x1 = np.zeros((batch, 224, 224, 3)) #gaus pyr 0
    batch_img_x2 = np.zeros((batch, 224, 224, 3)) #gaus pyr 1
    batch_img_x3 = np.zeros((batch, 224, 224, 3)) #gaus pyr 2

    batch_img_y = np.zeros((batch, 224, 224, 3)) #ori image
    count = 0
    directory = "F:/Dogs vs Cats Redux Kernels Edition/cat_dog_train/train/*.jpg"
    for num in glob.glob(directory):
        x1,x2,x3,y = my_imread_pyr(num) #Read image pyramid
        batch_img_x1[count] = x1
        batch_img_x2[count] = x2
        batch_img_x3[count] = x3

        batch_img_y[count] = y
        count += 1
        if count >= batch:
            count = 0
            print("ready to train!!!")
            model.fit(
                [batch_img_x1,batch_img_x2,batch_img_x3],
                batch_img_y,
                epochs=1,
                shuffle=None,
                verbose=1,
                batch_size=10)
            model.save(name)




