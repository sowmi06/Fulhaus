
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os



# loading Sun397 dataset
def Read_Dataset(root, SIZE):
    x = []
    y = []
    # reading file from parent path
    for parentPath, subdirs, files in os.walk(root):
        for subdir in subdirs:
            path = parentPath + "/" + subdir
            # label = subdir
            datafile = os.listdir(path)
            for file_name in datafile:
                imgPath = path + '/' + file_name
                # loading the images
                img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
                image = cv2.resize(img, (SIZE, SIZE))
                x.append(image)
                y.append(subdir)
                # printing the image shape and its labels
                print(img.shape)
                print(subdir)
    x = np.asarray(x)
    y = np.asarray(y)
    # using label encoder for the y_train and y_test
    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)
    return x, y
