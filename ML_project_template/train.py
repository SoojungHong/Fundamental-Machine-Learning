import json
import os
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from GarmentClassifier import GarmentClassifier

writer = SummaryWriter()

parser = argparse.ArgumentParser()
#parser.add_argument('data_dir')
#parser.add_argument('result_dir')

"""
main function
"""


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_example_data(training_loader, classes):
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j]] for j in range(4)))


def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


#def main(args):
def main():
    """
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:0')
    """

    """
    # create result_dir and its subdirectory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    # show 4 example images
    show_example_data(training_loader, classes)

    # model
    model = GarmentClassifier()

    """
    # Loss function
    """
    loss_fn = torch.nn.CrossEntropyLoss()

    # NB: Loss functions expect data in batches, so we're creating batches of 4
    # Represents the model's confidence in each of the 10 classes for a given input
    dummy_outputs = torch.rand(4, 10)
    # Represents the correct class among the 10 being tested
    dummy_labels = torch.tensor([1, 5, 3, 7])

    print(dummy_outputs)
    print(dummy_labels)

    loss = loss_fn(dummy_outputs, dummy_labels)
    print('Total loss for this batch: {}'.format(loss.item()))

    """
    Optimizer
    """
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    """
    Training loop
    """
    # Per Epoch Activity
    #train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn)
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_loader, optimizer, model, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


"""
execution entry
"""
if __name__ == '__main__':
    # if you want to pass the arguments
    #args = parser.parse_args()
    #args.data_dir = os.path.expanduser(args.data_dir)
    #args.result_dir = os.path.expanduser(args.result_dir)
    #main(args)

    main()