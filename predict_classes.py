import cv2
import numpy as np
from pathlib import Path
import time
from keras.models import load_model
test_sorted_dir = Path("./test2")
test_sorted_data = test_sorted_dir.glob("*.jpg")

images_names = []
net = load_model("google_model.h5")

for filename in test_sorted_data:
    images_names.append(str(filename))

for image in images_names:

    img = cv2.imread(image)
    img_norm = np.array(img, dtype=np.float32)
    img_norm = img_norm / 255.0
    score = [0, 0, 0]
    for i in range(4):
        for j in range(4):
            img_test_64x64 = img_norm[64 * i:64 * (i + 1), 64 * j:64 * (j + 1)]
            score = net.predict(np.reshape(a=img_test_64x64, newshape=(1, 64, 64, 3)))
            print(score)
            cv2.imshow("test", img_test_64x64)
            cv2.waitKey(25)
            time.sleep(5)


