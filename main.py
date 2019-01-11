import os
import random
import torch.utils.data
import dataset
import matplotlib.pyplot as plt
import numpy as np
import alexnet as al
import LFWNet
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from LFWNet import LfwNet

# make a data set . extract file paths and landmark feature cordonates from LFW train annotations
train_data_list = []
lfw_dataset_path = '/home/vishnusanjay/PycharmProjects/FacialReg/lfw'
test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
train_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_train.txt')
training_ratio = 0.8
aug_types = ['001', '010', '100', '011', '101', '110', '111']  # nnn - random crop , flip,brightness

training_validation_data_list = []
testing_data_list = []
learning_rate = 0.005

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = al.AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

if __name__ == '__main__':
    with open(train_landmark_path, "r") as file:
        for line in file:
            # split at tabs to get file name , borderbox co-ordinates , landmark feature co-ordinates
            tokens = line.split('\t')
            if len(tokens) == 3:
                file_path = tokens[0]
                crops = tokens[1].split()
                landmarks = tokens[2].split()
                training_validation_data_list.append(
                    {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': '000'})
                # random number of augmented images of original images
                # augtype 000 indicates original image
                random.shuffle(aug_types)
                max_augs = 5
                itr = 0
                for idx in aug_types:
                    training_validation_data_list.append(
                        {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': idx})
                    itr = itr + 1
                    if itr == max_augs:
                        break

                        # Read test data


    random.shuffle(training_validation_data_list)
    random.shuffle(testing_data_list)
    total_training_validation_items = len(training_validation_data_list)

    # Training dataset.
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = training_validation_data_list[: int(n_train_sets)]

    # Validation dataset.
    n_valid_sets = (1 - training_ratio) * total_training_validation_items
    valid_set_list = training_validation_data_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]

    # Testing dataset.
    test_set_list = testing_data_list

    train_dataset = dataset.LFWDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))

    valid_set = dataset.LFWDataset(valid_set_list)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=0)
    print('Total validation set:', len(valid_set))

    # Prepare pretrained model.
    alex_net = alexnet(pretrained=True)
    lfw_net = LfwNet()
    alex_dict = alex_net.state_dict()
    lfw_dict = lfw_net.state_dict()

    # Remove FC layers from pretrained model.
    alex_dict.pop('classifier.1.weight')
    alex_dict.pop('classifier.1.bias')
    alex_dict.pop('classifier.4.weight')
    alex_dict.pop('classifier.4.bias')
    alex_dict.pop('classifier.6.weight')
    alex_dict.pop('classifier.6.bias')

    # Load lfw model with pretrained data.
    lfw_dict.update(alex_dict)
    lfw_net.load_state_dict(lfw_dict)

    # Losses collection, used for monitoring over-fit
    train_losses = []
    valid_losses = []

    max_epochs = 2
    itr = 0
    optimizer = torch.optim.Adam(lfw_net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(lfw_net)

    for param in lfw_net.features.parameters():
        param.requires_grad = False


    print("start train")

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            itr += 1
            lfw_net.train()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())  # Use Variable(*) to allow gradient flow
            train_out = lfw_net.forward(train_input)  # Forward once

            # Compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)

            # Do the backward and compute gradients
            loss.backward()

            # Update the parameters with SGD
            optimizer.step()

            train_losses.append((itr, loss.item()))

            if train_batch_idx % 50 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

                # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
                lfw_net.eval()  # [Important!] set the network in evaluation model
                valid_loss_set = []  # collect the validation losses
                valid_itr = 0

                # Do validation
                for valid_batch_idx, (valid_input, valid_label) in enumerate(valid_data_loader):
                    lfw_net.eval()
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = lfw_net.forward(valid_input)  # forward once

                    # Compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, float(avg_valid_loss)))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)
    train_losses[0] =train_losses[1]
    valid_losses[0] =valid_losses[1]

    print(valid_losses)

    plt.plot(train_losses[2:, 0],  # Iteration
             train_losses[2:, 1])  # Loss value
    plt.plot(valid_losses[2:, 0],  # Iteration
             valid_losses[2:, 1])  # Loss value
    plt.show()
    net_state = lfw_net.state_dict()  # serialize trained model
    torch.save(net_state, os.path.join(lfw_dataset_path, 'lfwnet_1.pth'))  # save to disk