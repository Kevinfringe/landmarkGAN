import os
import cv2
import numpy as np
import pandas as pd

train_set_path = "../jaffedbase_official/train_set_jpg/"
lm_csv_path = "../jaffedbase_official/lm_csv/"
lm_img_path = "../jaffedbase_official/lm_img/"

# # Test csv process.
# df = pd.read_csv(os.path.join(lm_genuine_path, "1_GA1.csv"))
# print(df)
# df_array = df.to_numpy()
# print(df_array.shape)
#
# # Test imread
# img = cv2.imread(os.path.join(genuine_img_path, "1_GA1.jpg"))
# print(img[0:9,0:9].shape)
# temp = np.ones_like(img) * 255
# print(temp)

# for file in os.listdir(lm_csv_path):
#     if file.endswith(".txt"):
#         os.remove(os.path.join(lm_csv_path, file))
#         print("Clean target file!")


# Iterate corresponding image, and generate landmark image.
# Specifically, pixels around the landmark coordinates within radius of
# 4 is filled with the corresponding pixel values in original image, and
# white color elsewhere.
def lmImageGenerate(img_path, csv_path, save_path, radius):
    """
        Generate the landmark image, and save in corresponding path.
    :param save_path: path to save generated landmark images.
    :param img_path: input image path.
    :param csv_path: csv files that record landmarks of each image.
    :return: None
    """
    for file in os.listdir(img_path):
        filename = file.strip('.jpg')
        print(filename)

        # read the lm csv file as np.array.
        df = pd.read_csv(os.path.join(csv_path, filename + ".csv"))
        lm_array = df.to_numpy()
        print(lm_array.shape)
        # read the img file via opencv.
        img = cv2.imread(os.path.join(img_path, file))
        width = img.shape[1]
        height = img.shape[0]

        # Initialize the variable to store output.
        temp = np.ones_like(img) * 255.0
        lm_length = (lm_array.shape[1] - 2) // 2
        print("The length of lm_array is " + str(lm_length))

        # Since the landmark coordinates in lm_array are ordered as
        # [x1 ... xn, y1 ... yn].
        for i in range(lm_length):
            x_cor = int(lm_array[:, 2 + i]) if (lm_array[:, 2 + i] < width - radius) else width - radius
            x_cor = x_cor if x_cor > radius else radius
            y_cor = int(lm_array[:, 2 + lm_length + i]) if lm_array[:, 2 + lm_length + i] < height - radius else height - radius
            y_cor = y_cor if y_cor > radius else radius

            # Fill the pixels around landmark with corresponding RGB values as
            # original image.
            temp[y_cor - radius: y_cor + radius, x_cor - radius: x_cor + radius] = img[y_cor - radius: y_cor + radius, x_cor - radius: x_cor + radius]
            # update the original csv files at the mean time.
            df.iloc[0, 2+i] = x_cor
            df.iloc[0, 2+lm_length+i] = y_cor

        # save the img to save_path
        target_path = os.path.join(save_path, file)
        cv2.imwrite(target_path, temp)
        # Overwrite the original csv file.
        df.to_csv(os.path.join(csv_path, filename + ".csv"), index=False)

# Test
lmImageGenerate(train_set_path, lm_csv_path, lm_img_path, 1)