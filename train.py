import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from keras import utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from model_arch import *


def get_data_frame():
    image_labels = []
    plastic_dir = Path("./plastic64x64/")
    paper_dir = Path("./paper64x64/")
    metal_dir = Path("./metal64x64/")

    plastic_data = plastic_dir.glob('*.jpg')
    paper_data = paper_dir.glob('*.jpg')
    metal_data = metal_dir.glob('*.jpg')

    for file in plastic_data:
        image_labels.append((file, 0))  # [1, 0, 0] plastic
    for file in paper_data:
        image_labels.append((file, 1))  # [0, 1, 0] paper
    for file in metal_data:
        image_labels.append((file, 2))  # [0, 0, 1] metal

    df = pd.DataFrame(image_labels, columns=['Image', 'Label'], index=None)
    s = np.arange(df.shape[0])
    np.random.shuffle(s)
    df = df.iloc[s, :].reset_index(drop=True)
    return df


def transform_batch(img_path_list, labels_list):
    img_list = []
    for i in range(len(img_path_list)):
        img = cv2.imread(str(img_path_list[i]), cv2.IMREAD_COLOR)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img_list.append(img)

    return np.array(img_list), labels_list


df_data = get_data_frame()
labels = np.array(df_data.iloc[:, 1]).reshape((df_data.shape[0], 1))
data, _ = transform_batch(df_data.iloc[:, 0], df_data.iloc[:, 1])
labels = utils.to_categorical(labels, 3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
del data, labels

# CNN Parameters


net = CnnCreator.wastes_v1_arch(input_shape, kernel_size, pool_size, filters_base)

opt = Adam(lr=0.0001, epsilon=bnEps, decay=0.00001, amsgrad=True)
net.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])


aug = ImageDataGenerator(rotation_range=35, zoom_range=0.3, width_shift_range=0.4, height_shift_range=0.4,
                         shear_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode="wrap")

aug.fit(X_train)
history = net.fit_generator(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=70,
                            shuffle=True)
net.save("goole_net_v2.h")