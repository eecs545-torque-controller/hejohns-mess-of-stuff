#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas
import numpy as np
import sys
import os
import re
import time
import copy
# our files
from config import *
import dataloader

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=4):
        super().__init__()
        self.input_size = len(sensor_list)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=self.input_size, # number of expected features in input
                # TODO: trial and error
                hidden_size=hidden_size, # number of features in hidden state
                # TODO: trial and error
                num_layers=num_layers, # number of recurrent layers
                batch_first=True,
                )
        self.linear = nn.Linear(hidden_size, len(output_list))
        #self.dropout = nn.Dropout(p=0.4) # paper used p=0.4
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        if LAST: # if we're only interested in the moments at the last timestamp
            x = x[:, -1, :]
        return x

# my brain needs to see these as globals
if __name__ == '__main__':
    # gpu stuff
    print(f"Running on {device}")
    # args
    assert len(sys.argv) > 1
    window_size = get_window_size(sys.argv[1])
    use_greedy = window_size < 50
    print(f"use_greedy is {use_greedy}")
    grandUnifiedData, windows, normalization_params = read_entire_pickle(sys.argv[1])
    # basic initialization
    model = LSTMModel()
    model = nn.DataParallel(model)
    model = model.to(device, non_blocking=True)
    if SCHEDULER:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001) # taken from the paper
    loss_fn = nn.MSELoss() # taken from the paper

    if DEBUG:
        subjects = ['AB01', 'AB02', 'AB05']
    else:
        subjects = grandUnifiedData.keys()
    test_subjects = ['AB01']
    validation_subjects = ['AB05']
    training_subjects = [s for s in subjects if s not in test_subjects and s not in validation_subjects]
    if DEBUG:
        activities = re.compile("normal_walk_1_0-6"); # smaller dataset
    else:
        activities = re.compile(".");
    print(f"initializing training dataset... {curtime()}")
    # error checking
    num_total_windows = len(windows)
    if use_greedy:
        training_data = dataloader.GreedyGrandLSTMDataset(window_size, (grandUnifiedData, windows), training_subjects, activities)
    else:
        training_data = dataloader.GrandLSTMDataset(window_size, (grandUnifiedData, windows), training_subjects, activities)
    num_training_windows = training_data.__len__()
    print(f"done initializing training dataset... {curtime()}")
    print(f"initializing validation dataset... {curtime()}")
    if use_greedy:
        # NOTE: training, validation, and test data should never overlap, so we can reuse the dict and list
        if training_data.unused_data:
            grandUnifiedData = training_data.unused_data
        validation_data = dataloader.GreedyGrandLSTMDataset(window_size, (grandUnifiedData, windows), validation_subjects, activities)
    else:
        validation_data = dataloader.GrandLSTMDataset(window_size, (grandUnifiedData, windows), validation_subjects, activities)
    num_validation_windows = validation_data.__len__()
    print(f"done initializing validation dataset... {curtime()}")
    print(f"initializing test dataset... {curtime()}")
    if use_greedy:
        # NOTE: training, validation, and test data should never overlap, so we can reuse the dict and list
        if validation_data.unused_data:
            grandUnifiedData = validation_data.unused_data
        test_data = dataloader.GreedyGrandLSTMDataset(window_size, (grandUnifiedData, windows), test_subjects, activities)
        del grandUnifiedData # drop the reference, if it's the last one (eg using GreedyGrandLSTMDataset)
    else:
        test_data = dataloader.GrandLSTMDataset(window_size, (grandUnifiedData, windows), test_subjects, activities)
    num_test_windows = test_data.__len__()
    print(f"done initializing test dataset... {curtime()}")
    # NOTE: if we're using all the data
    if not num_total_windows == num_training_windows + num_validation_windows + num_test_windows:
        print("!!!We must not be using all the data!!!")
        assert DEBUG

# for y_lamba
def drop_for_just_final(y_pred, y_batch):
    y_pred = y_pred[:, -1, :]
    y_batch = y_batch[:, -1, :]
    return y_pred, y_batch

def loop_over_data(
        dataloader,
        loss_fn,
        batch_loss_lambda=lambda bl, _: bl,
        y_lambda=lambda x, y: (x, y),
        optimizer=None
        ):
    total_loss = 0.0
    num_samples = 0
    num_elements = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        y_pred = model(X_batch)
        assert not y_pred.isnan().any()

        y_pred, y_batch= y_lambda(y_pred, y_batch)
        batch_loss = loss_fn(y_pred, y_batch)
        assert not batch_loss.isnan().any()

        if optimizer is not None:
            optimizer.zero_grad() # this can go anywhere except for between backward and step
            batch_loss.backward()
            optimizer.step()

        numel = y_batch.numel()
        batch_loss = batch_loss.sum().item()
        batch_loss = batch_loss_lambda(batch_loss, numel)
        total_loss += batch_loss # sum of a singleton is id
        num_samples += X_batch.size()[0] # batch_size
        num_elements += numel
    assert num_samples == dataloader.dataset.__len__()
    return total_loss, num_elements

# How MSELoss is calculated, which we'll have to do by hand over the batches
# https://discuss.pytorch.org/t/how-is-the-mseloss-implemented/12972/4
# https://discuss.pytorch.org/t/custom-loss-functions/29387/2

# just_final means that we only consider the moments at the last timestamp
def total_mse(dataloader, just_final):
    assert not model.training
    # just_final is irrespective if LAST is set-- this WILL NOT WORK if LAST with just_final = True and LAST is already set
    assert not (just_final and LAST)
    total_loss, num_elements = 0.0, 0
    if just_final:
        total_loss, num_elements = loop_over_data(dataloader, loss_fn=nn.MSELoss(reduction="none"), y_lambda=drop_for_just_final)
    else:
        total_loss, num_elements = loop_over_data(dataloader, loss_fn=nn.MSELoss(reduction="none"))
    return total_loss, num_elements

def eval_rmse(dataloader, just_final):
    assert not model.training
    total_loss, num_elements = total_mse(dataloader, just_final)
    error = rmse(total_loss, num_elements)
    return error

def rmse(total_loss, num_samples):
    return np.sqrt(total_loss / num_samples)

# https://stackoverflow.com/a/73704579
class EarlyStop:
    def __init__(self, patience=50): # paper used patience=50
        self.count = 0
        self.min_loss = float('inf')
        self.patience = patience
    # returns true iff we should stop early
    def should_early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.count = 0
        elif loss > self.min_loss:
            self.count += 1
            if self.count > self.patience:
                return True
        return False

# for local testing
#torch.set_num_threads(48)

if __name__ == '__main__':
    # I'm pretty sure prefetching is useless if we're doing CPU training
    # unless the disk IO is really slow, but I'm hoping for gpu we can make
    # better use of both resources
    use_workers = device == "cuda" and not use_greedy
    train_dataloader = torch.utils.data.DataLoader(
            training_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=(device == "cuda"),
            num_workers=(4 if use_workers else 0),
            persistent_workers=use_workers,
            )
    validation_dataloader = torch.utils.data.DataLoader(
            validation_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=(device == "cuda"),
            num_workers=(2 if use_workers else 0),
            persistent_workers=use_workers,
            )
    test_dataloader = torch.utils.data.DataLoader(
            test_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=(device == "cuda"),
            num_workers=(2 if use_workers else 0),
            persistent_workers=use_workers,
            )

    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        checkpoint = torch.load(sys.argv[2])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        print(f"Checkpoint {sys.argv[1]} loaded. Resuming training from epoch {start_epoch}... {curtime()}")
    else:
        start_epoch = 0
        print(f"No checkpoint found since {len(sys.argv) > 2}.")
        print(f"Starting training from scratch.... {curtime()}")

    should_early_stop = EarlyStop() # taken from the paper
    for epoch in range(start_epoch, n_epochs):
        print(f"epoch {epoch} at {curtime()}", flush=True)
        model.train()
        total_training_loss, num_elements = loop_over_data(
                train_dataloader,
                loss_fn=loss_fn,
                batch_loss_lambda=lambda bl, numel: bl * numel,
                optimizer=optimizer
                )
        # save checkpoint every epoch

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss
            }, "saved_model." + f"{window_size}." + str(epoch + 1) + ".ckpt")

        # eval every epoch
        model.eval()
        with torch.no_grad():
            if LAST:
                train_rmse = total_training_loss / num_elements
                validation_rmse = eval_rmse(validation_dataloader, False)
                test_rmse = eval_rmse(test_dataloader, False)
                print("Epoch %d: final timestamp: train RMSE %.4f, validation RMSE %.4f, test RMSE %.4f"% (epoch, train_rmse, validation_rmse, test_rmse))
                val_rmse = validation_rmse
            else:
                train_rmse = total_training_loss / num_elements
                train_rmse_just_final = eval_rmse(train_dataloader, True)
                validation_rmse = eval_rmse(validation_dataloader, False)
                validation_rmse_just_final = eval_rmse(validation_dataloader, True)
                test_rmse = eval_rmse(test_dataloader, False)
                test_rmse_just_final = eval_rmse(test_dataloader, True)
                print("Epoch %d: whole window: train RMSE %.4f, validation RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, validation_rmse, test_rmse))
                print("Epoch %d: final timestamp: train RMSE %.4f, validation RMSE %.4f, test RMSE %.4f"% (epoch, train_rmse_just_final, validation_rmse, test_rmse_just_final))
                val_rmse = validation_rmse_just_final
            if SCHEDULER:
                scheduler.step(val_rmse)
                print("Current scheduler learning rate for epoch %d is %.4f" % (epoch, scheduler.get_last_lr()[0]))
                if scheduler.get_last_lr()[0] < .000001:
                    print(f" Training rate too low on epoch {epoch}")
                    break

        # TODO: this should really be validation, but we're not really using it anyways
        # I've temporarily made it the val_rmse
        if should_early_stop.should_early_stop(val_rmse):
            print(f"Stopping early on epoch {epoch}")
            break
    print(f"Finished Training at {curtime()}", flush=True)
