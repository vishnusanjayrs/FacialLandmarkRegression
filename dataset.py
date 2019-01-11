import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import main


class LFWDataset(Dataset):
    img_w, img_h = 225.0, 225.0
    dir_name_trim_length = -9
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # split data from the dictionary into class variables
        file_name = item['file_path']
        # print(file_name)
        self.file_path = os.path.join(main.lfw_dataset_path, file_name[:self.dir_name_trim_length], file_name)
        # print(self.file_path)
        self.crops = np.asarray(item['crops'])
        self.crops = self.crops.astype(int)
        self.landmarks = np.asarray(item['landmarks'])
        self.landmarks = self.landmarks.astype(float)
        self.aug_types = item['aug_types']
        self.aug_type_list = list(self.aug_types)
        # open image from the file path
        orig_img = Image.open(self.file_path)

        # prepare image based on input augmentation type

        # crop or random crop the image to extract face using crop co-ordinates
        img, landmarks = self.crop(orig_img)

        # flip the image based on condition
        if self.aug_type_list[1] == '1':
            img, landmarks = self.flip(img, landmarks)

        # print(landmarks)

        # brighten the image based on condition
        if self.aug_type_list[2] == '1':
            img = self.brighten(img)

        # self.preview(img, landmarks)

        # normalise the image and landmarks
        img, landmarks = self.normalise(img, landmarks)

        # convert np.lsit to tensor
        # Create tensors and reshape them to proper sizes.
        img_tensor = torch.Tensor(img.astype(float))
        img_tensor = img_tensor.view((img.shape[2], img.shape[0], img.shape[1]))
        landmark_tensors = torch.Tensor(landmarks)
        # print(landmark_tensors.shape[0])

        # print("done")

        orig_img.close()


        # 225 x 225 x 3 input image tensor. 14 landmark tensor.

        return img_tensor, landmark_tensors

    def crop(self, input_img):
        crops_offset = np.zeros(len(self.crops), dtype=np.float32)
        if self.aug_type_list[0] == '1':
            # randomly add  a number between -5 and 5 to the crop co-ordinates
            for index in range(0, len(self.crops)):
                rand_offset = randint(-1, 1)
                crops_offset[index] = self.crops[index] + rand_offset
        else:
            crops_offset = self.crops

        # crop the image
        img = input_img.crop((crops_offset))
        # change landmark co-ordinates by subtracting cropped length from the landmark in each dimension
        landmarks_offset = np.zeros(2, dtype=np.float32)
        landmarks_offset = [crops_offset[0], crops_offset[1]]
        landmarks_offset = np.tile(landmarks_offset, 7)
        cropped_landmarks = self.landmarks - landmarks_offset
        # resize the image to (225,225) which is the input image size to alexnet
        w, h = img.size
        # find the image size ratio to 225 so that same ratio can be applies to landmark co-ordinates
        ratio_width = w / self.img_w
        ratio_height = h / self.img_h
        img = img.resize((225, 225), Image.ANTIALIAS)
        landmark_offset_ratio = [ratio_width, ratio_height]
        landmark_offset_ratio = np.tile(landmark_offset_ratio, 7)
        cropped_landmarks = cropped_landmarks / landmark_offset_ratio
        return img, cropped_landmarks

    def flip(self, input_img, input_landmarks):
        # flip the image
        img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
        # flip the x co-ordinates in the landmarks list
        flipped_landmarks = np.zeros(len(self.landmarks), dtype=np.float32)
        for index in range(0, len(self.landmarks)):
            if index % 2 == 0:
                flipped_landmarks[index] = 225.0 - input_landmarks[index]
            else:
                flipped_landmarks[index] = input_landmarks[index]

        # tranform the landmark co ordinates to keep the co-ordinates side consistensy
        transformed_landmarks = np.zeros(14, dtype=np.float32)

        transformed_landmarks[0] = flipped_landmarks[6]
        transformed_landmarks[1] = flipped_landmarks[7]
        transformed_landmarks[2] = flipped_landmarks[4]
        transformed_landmarks[3] = flipped_landmarks[5]
        transformed_landmarks[4] = flipped_landmarks[2]
        transformed_landmarks[5] = flipped_landmarks[3]
        transformed_landmarks[6] = flipped_landmarks[0]
        transformed_landmarks[7] = flipped_landmarks[1]
        transformed_landmarks[8] = flipped_landmarks[10]
        transformed_landmarks[9] = flipped_landmarks[11]
        transformed_landmarks[10] = flipped_landmarks[8]
        transformed_landmarks[11] = flipped_landmarks[9]
        transformed_landmarks[12] = flipped_landmarks[12]
        transformed_landmarks[13] = flipped_landmarks[13]

        return img, transformed_landmarks

    def brighten(self, input_img):
        # randonly initialize brightening/darkening factor
        factor = 1 + randint(-30, 30) / 100.0
        brightened_img = input_img.point(lambda x: x * factor)
        return brightened_img

    def normalise(self, input_img, input_landmarks):
        # convert image to array
        n_img = np.asarray(input_img, dtype=np.float32)
        # normalise the image pixels to (-1,1)
        img = (n_img / 255.0) * 2 - 1

        # normalise landmarks
        n_landmarks = input_landmarks / self.img_w
        return n_img, n_landmarks

    def denormalize(self, input_img, input_landmarks):
        # Denormalize the image.
        dn_img = np.array(input_img, dtype=float)
        if dn_img.shape[0] == 3:
            dn_img = (dn_img + 1) / 2 * 255
            return dn_img.astype(int)

        # Denormalize the landmarks.
        return input_landmarks * self.img_w

    def preview(self, image, landmarks):

        # image, landmarks = self.denormalize(image_tensor, landmark_tensor)

        plt.figure(num='Preview')
        plt.imshow(image)

        plt.scatter(x=landmarks[0], y=landmarks[1], c='r', s=10)
        plt.scatter(x=landmarks[2], y=landmarks[3], c='b', s=10)
        plt.scatter(x=landmarks[4], y=landmarks[5], c='g', s=10)
        plt.scatter(x=landmarks[6], y=landmarks[7], c='c', s=10)
        plt.scatter(x=landmarks[8], y=landmarks[9], c='m', s=10)
        plt.scatter(x=landmarks[10], y=landmarks[11], c='y', s=10)
        plt.scatter(x=landmarks[12], y=landmarks[13], c='k', s=10)
        plt.xlim(0, 225)
        plt.ylim(225, 0)
        plt.show()
