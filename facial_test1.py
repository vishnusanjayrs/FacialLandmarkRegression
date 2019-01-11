import numpy as np
import main
import dataset
import LFWNet
import alexnet
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
from random import randint
from torch.autograd import Variable

n_detection_range = 100
n_size = 128

if __name__ == '__main__':
    lfw_dataset_path = '/home/vishnusanjay/PycharmProjects/FacialReg/lfw'
    test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
    testing_data_list = []

    with open(test_landmark_path, "r") as file:
        for line in file:
            # split at tabs to get file name , borderbox co-ordinates , landmark feature co-ordinates
            tokens = line.split('\t')
            if len(tokens) == 3:
                file_path = tokens[0]
                crops = tokens[1].split()
                landmarks = tokens[2].split()
                testing_data_list.append(
                    {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': '000'})

    random.shuffle(testing_data_list)

    # Testing dataset.
    test_set_list = testing_data_list

    test_dataset = dataset.LFWDataset(test_set_list)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total teting items', len(test_dataset), ', Total training batches per epoch:', len(test_data_loader))

    test_net = LFWNet.LfwNet()

    test_net_state = torch.load(os.path.join(lfw_dataset_path, 'lfwnet_1.pth'))

    test_net.load_state_dict(test_net_state)

    # Generate predictions.
    loss_raw = []
    for test_batch_idx, (test_input, test_oracle) in enumerate(test_data_loader):
        test_net.eval()
        test_net.cuda()
        predictions = test_net.forward(test_input.cuda())



        # Compute overall loss.
        pred_array = np.array(predictions.detach())
        pred_array = pred_array.reshape(7,2)

        #print(pred_array*225)
        test_array = np.array(test_oracle.detach())
        test_array = test_array.reshape(7,2)
        #print(test_array * 225)
        batch_loss = np.linalg.norm(pred_array-test_array, axis=1)
        #print(np.array(predictions.detach()).shape)
        #print(np.array(test_oracle.detach()).shape)
        #print(batch_loss*225)
        batch_loss *= 225
        loss_raw.extend(batch_loss.tolist())
       # print(loss_raw.shape)




    # Compute loss plot.
    loss_raw = np.array(loss_raw).flatten()
    #print(loss_raw.shape)
    loss_plot = []
    for step in range(1, n_detection_range):
        loss_plot.append((step, len(np.where(loss_raw < step)[0]) / float(len(loss_raw))))
        #print(len(np.where(loss_raw < step)[0]))
        #print(len(loss_raw))
        #print(len(np.where(loss_raw < step)[0]) / len(loss_raw))
    loss_plot = np.asarray(loss_plot,dtype=np.float32)
    print(loss_plot)
    plt.figure(num='Percentage of Detected Key-points')
    plt.title('Percentage of Detected Key-points')
    plt.xlabel('L2 Distance From Detected Points to Ground Truth Points')
    plt.ylabel('Percentage')
    axes = plt.gca()
    axes.set_xticks(np.arange(0, n_detection_range, 5))
    axes.set_yticks(np.arange(-0.1, 1.1, 0.1))
    axes.set_xlim([0, n_detection_range])
    axes.set_ylim([-0.05, 1.05])
    plt.grid()
    plt.plot(loss_plot[:, 0], loss_plot[:, 1])

    plt.show()

