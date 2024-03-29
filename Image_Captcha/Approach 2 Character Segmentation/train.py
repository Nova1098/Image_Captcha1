# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 03:32:26 2023

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import CharactersDataset
from model import CharacterClassifier
from torch.utils.tensorboard import SummaryWriter
from train_utils import *

base_path = r'/users/pgrad/singhr6'
cwd = r'/users/pgrad/singhr6/model_training'
os.chdir(cwd)

# constants
VALIDATION_BATCH_SIZE = 32

# save Tensorboard logs somewhere
log_dir = "logs/lr_{}_batchsize_{}".format(args.learning_rate, args.batch_size)
writer = SummaryWriter(log_dir)

# we need to save the models somewhere
model_fn = "epoch_{}_lr_{}_batchsize_{}"
if not os.path.exists("models"):
    os.mkdir("models")

# are we using GPU or CPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Cuda available? {}".format(torch.cuda.is_available()))
print("Device: {}".format(device))

# initialize model, dataset, optimizers, and data loaders
model = CharacterClassifier(num_classes = 94, pretrained = False).to(device)

model_path = "/users/pgrad/singhr6/model_training/models/epoch_0_lr_5e-05_batchsize_32"
if model_path:
    model.load_state_dict(torch.load(model_path))
    model.train()

train_dataset = CharactersDataset(data_root = os.path.join(base_path, "segment_data" ,"data", "characters", "train"), validate = False)
val_dataset = CharactersDataset(data_root = os.path.join(base_path, "segment_data", "data", "characters", "val"), validate = True)
#train_dataset = CharactersDataset(data_root = os.path.join(base_path, "Desktop", "ProcessedTrainImage2"), validate = False)
#val_dataset = CharactersDataset(data_root = os.path.join(base_path, "Desktop", "ProcessedValidateImage2"), validate = True)

print("\nTrain dataset size: {}".format(len(train_dataset)))
print("Validation dataset size: {}".format(len(val_dataset)))

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay, betas = (0.9, 0.999))
loss_func = nn.CrossEntropyLoss(reduction = 'sum').to(device)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 1)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = VALIDATION_BATCH_SIZE, shuffle = False, num_workers = 1)

#lr_lambda = lambda epoch : epoch*0.95
#lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# train and validate the model
print("\nTraining with settings: ")
print("\tLearning rate: ", args.learning_rate)
print("\tWeight decay: ", args.weight_decay)
print("\tBatch size: ", args.batch_size)
print("\tEpochs: ", args.num_epochs)

char_to_mask = {
    '<': '_101_',
    '>': '_102_',
    ':': '_103_',
    '"': '_104_',
    '/': '_105_',
    '\\': '_106_',
    '|': '_107_',
    '?': '_108_',
    '*': '_109_',
    '.': '_111_'
}

def encode_special_characters(filename):
    for char, mask in char_to_mask.items():
        filename = filename.replace(char, mask)
    return filename

def decode_special_characters(encoded_filename):
    for char, mask in char_to_mask.items():
        encoded_filename = encoded_filename.replace(mask, char)
    return encoded_filename

highest_validation_accuracy = None
highest_validation_epoch = None

for epoch in range(args.num_epochs):
    # train the model
    for batch_id, samples in enumerate(train_loader):
        labels = samples['labels'].to(device, dtype = torch.int64)
        
        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)

        loss = loss_func(pred, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Training loss', loss.item(), epoch * len(train_loader) + batch_id)
        if batch_id % 50 == 0:
            print("(train) => Epoch {}/{} - Batch {}/{} - Loss: {}".format(epoch, args.num_epochs, batch_id, len(train_loader), loss.item()))
    #lr_scheduler.step()
    
    # validate the model
    validation_loss = 0.0
    correct = 0
    for batch_id, samples in enumerate(val_loader):
        labels = samples['labels'].to(device, dtype = torch.int64)

        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)

        pred_label = torch.argmax(pred, 1) if VALIDATION_BATCH_SIZE > 1 else torch.argmax(pred)
        correct += torch.sum(pred_label == labels).item() if VALIDATION_BATCH_SIZE > 1 else (1 if pred_label == labels[0] else 0)

        loss = loss_func(pred, labels)
        validation_loss += loss.item()

        if batch_id % 50 == 0:
            print("(validation) => Epoch {}/{} - Batch {}/{} - Loss: {}, Running loss: {}".format(epoch, args.num_epochs, batch_id, len(val_loader), loss.item(), validation_loss))
    validation_loss /= len(val_loader)
    validation_accuracy = correct / len(val_dataset)
    print("(validation) => Epoch {}/{} - Batch {}/{} - Avg. Loss: {}".format(epoch, args.num_epochs, batch_id, len(val_loader), validation_loss))
    print("Validation accuracy: {}% (Highest = {})".format(round(validation_accuracy * 100.0, 2), 0 if highest_validation_accuracy is None else (highest_validation_accuracy * 100.0)))
    writer.add_scalar('Validation accuracy', validation_accuracy, epoch)
    writer.add_scalar('Validation loss', validation_loss, epoch)

    # we found a new 'best' model in terms of validation accuracy, so save it to disk
    highest_validation_accuracy_string = "None" if highest_validation_accuracy is None else round(highest_validation_accuracy, 5)
    if highest_validation_accuracy is None or validation_accuracy > highest_validation_accuracy:
        # remove the old 'best' model
        if highest_validation_epoch is not None:
            old_model_path = os.path.join("models", model_fn.format(highest_validation_epoch, args.learning_rate, args.batch_size))
            if os.path.exists(old_model_path):
                os.remove(old_model_path)

        # save new best model
        print("** New best model ; Epoch {} ; Old best val accuracy: {}, New best val accuracy: {}".format(epoch, highest_validation_accuracy_string, round(validation_accuracy, 5)))
        torch.save(model.state_dict(), os.path.join("models", model_fn.format(epoch, args.learning_rate, args.batch_size)))
        highest_validation_accuracy = validation_accuracy
        highest_validation_epoch = epoch

