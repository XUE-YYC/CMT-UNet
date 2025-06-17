import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing
from PIL import Image
# CLASS = ['background','car','tree','low vegetation','building','impervious surfaces']

def crop(img,target,size,overlap,train):
    img_name = img.split('/')[-1].split('.')[0]
    image = cv2.imread(img)
    target = cv2.imread(target)
    target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
    assert image.shape[:2]==target.shape[:2]
    number = 0
    for i in range((image.shape[0]-size)//overlap+1):
        for j in range((image.shape[1]-size)//overlap+1):
            image_ = image[i*overlap:i*overlap+size,j*overlap:j*overlap+size,:]
            target_ = target[i*overlap:i*overlap+size,j*overlap:j*overlap+size,:].reshape(-1,3)
            target_[(target_==[255,0,0]).all(axis=1)] = np.array([0])
            target_[(target_ == [255, 255, 0]).all(axis=1)] = np.array([1])
            target_[(target_ == [0, 255, 0]).all(axis=1)] = np.array([2])
            target_[(target_ == [0, 255, 255]).all(axis=1)] = np.array([3])
            target_[(target_ == [0, 0, 255]).all(axis=1)] = np.array([4])
            target_[(target_ == [255, 255, 255]).all(axis=1)] = np.array([5])
            target_ = target_[:,0]
            target_ = target_.reshape(image_.shape[0],image_.shape[1])
            if train:
                cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg' + '/' + img_name + str(number) + '.jpg', image_)
                cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target' + '/' + img_name + str(number) + '.png', target_)
            else:
                cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg_test' + '/' + img_name + str(number) + '.jpg', image_)
                cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target_test' + '/' + img_name + str(number) + '.png', target_)
            # print(r'F:\work\2022_9\Potsdam\dataset\jpg'+'//'+img_name+str(number)+'.jpg')
            number += 1

        image_ = image[i*overlap:i*overlap+size,-size:,:]
        target_ = target[i*overlap:i*overlap+size,-size:,:].reshape(-1, 3)
        target_[(target_ == [255, 0, 0]).all(axis=1)] = np.array([0])
        target_[(target_ == [255, 255, 0]).all(axis=1)] = np.array([1])
        target_[(target_ == [0, 255, 0]).all(axis=1)] = np.array([2])
        target_[(target_ == [0, 255, 255]).all(axis=1)] = np.array([3])
        target_[(target_ == [0, 0, 255]).all(axis=1)] = np.array([4])
        target_[(target_ == [255, 255, 255]).all(axis=1)] = np.array([5])
        target_ = target_[:, 0]
        target_ = target_.reshape(image_.shape[0], image_.shape[1])
        if train:
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg' + '/' + img_name + str(number) + '.jpg', image_)
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target' + '/' + img_name + str(number) + '.png', target_)
        else:
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg_test' + '/' + img_name + str(number) + '.jpg', image_)
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target_test' + '/' + img_name + str(number) + '.png', target_)
        number += 1
    for i in range((image.shape[1]-size)//overlap+1):
        image_ = image[-size:,i*overlap:i*overlap+size,:]
        target_ = target[-size:,i*overlap:i*overlap+size,:].reshape(-1, 3)
        target_[(target_ == [255, 0, 0]).all(axis=1)] = np.array([0])
        target_[(target_ == [255, 255, 0]).all(axis=1)] = np.array([1])
        target_[(target_ == [0, 255, 0]).all(axis=1)] = np.array([2])
        target_[(target_ == [0, 255, 255]).all(axis=1)] = np.array([3])
        target_[(target_ == [0, 0, 255]).all(axis=1)] = np.array([4])
        target_[(target_ == [255, 255, 255]).all(axis=1)] = np.array([5])
        target_ = target_[:, 0]
        target_ = target_.reshape(image_.shape[0], image_.shape[1])
        if train:
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg' + '/' + img_name + str(number) + '.jpg', image_)
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target' + '/' + img_name + str(number) + '.png', target_)
        else:
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg_test' + '/' + img_name + str(number) + '.jpg', image_)
            cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target_test' + '/' + img_name + str(number) + '.png', target_)
        number += 1

    image_ = image[-size:,-size:,:]
    target_ = target[-size:,-size:,:].reshape(-1, 3)
    target_[(target_ == [255, 0, 0]).all(axis=1)] = np.array([0])
    target_[(target_ == [255, 255, 0]).all(axis=1)] = np.array([1])
    target_[(target_ == [0, 255, 0]).all(axis=1)] = np.array([2])
    target_[(target_ == [0, 255, 255]).all(axis=1)] = np.array([3])
    target_[(target_ == [0, 0, 255]).all(axis=1)] = np.array([4])
    target_[(target_ == [255, 255, 255]).all(axis=1)] = np.array([5])
    target_ = target_[:, 0]
    target_ = target_.reshape(image_.shape[0], image_.shape[1])
    if train:
        cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg' + '/' + img_name + str(number) + '.jpg', image_)
        cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target' + '/' + img_name + str(number) + '.png', target_)
    else:
        cv2.imwrite(r'/home/Data_set/Potsdam/dataset/jpg_test' + '/' + img_name + str(number) + '.jpg', image_)
        cv2.imwrite(r'/home/Data_set/Potsdam/dataset/target_test' + '/' + img_name + str(number) + '.png', target_)
    number += 1

# img = img_file+'//'+'top_potsdam_2_10_RGB.tif'
# target = target_file+'//'+'top_potsdam_2_10_label.tif'
# crop(img,target,640,320)


if __name__ == '__main__':

    img_file = r'/home/Data_set/Potsdam/2_Ortho_RGB/2_Ortho_RGB'
    train_target_file = r'/home/Data_set/Potsdam/5_Labels_for_participants/5_Labels_for_participants'
    target_file = r'/home/Data_set/Potsdam/5_Labels_all'

    train_list = os.listdir(train_target_file)
    all_list = os.listdir(target_file)
    test_list = [i for i in all_list if i not in train_list]


    multiprocessing.freeze_support()
    multiprocessing.Process()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    SIZE = 640
    OVERLAP = 320

    for file in all_list:
        if file in train_list:
            TRAIN = True
        else:
            TRAIN = False
        IMG = img_file+'/'+file.replace('label','RGB')
        TARGET = target_file + '/' + file
        pool.apply_async(func=crop, args=[IMG,TARGET,SIZE,OVERLAP,TRAIN ])
    pool.close()
    pool.join()




