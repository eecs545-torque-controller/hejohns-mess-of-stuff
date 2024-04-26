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
# our files
from config import *
import dataloader

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=1):
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
        ## if we're only interested in the moments at the last timestamp
        #x = x[:, -1, :]
        return x
    def total_mse(dataloader, loss_fn):
        assert not model.training
        total_loss = 0.0
        num_samples = 0
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            assert not y_pred.isnan().any()
            batch_loss = loss_fn(y_pred, y_batch)
            assert not batch_loss.isnan().any()
            batch_size = X_batch.size()[0]
            total_training_loss += batch_loss.item() * batch_size
            num_training_batches += batch_size
        return total_loss, num_samples
    def total_mse_just_final(dataloader, loss_fn):
        assert not model.training
        total_loss = 0.0
        num_samples = 0
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            assert not y_pred.isnan().any()
            y_pred = y_pred[:, -1, :]
            y_batch = y_batch[:, -1, :]
            batch_loss = loss_fn(y_pred, y_batch)
            assert not batch_loss.isnan().any()
            total_loss += batch_loss.item()
            num_samples += 1
        return total_loss, num_samples
    def eval_rmse(dataloader, loss_fn):
        assert not model.training
        total_loss, num_samples = total_mse(dataloader, loss_fn)
        error = rmse(total_loss, num_samples)
        total_loss, num_samples = total_mse_just_final(dataloader, loss_fn)
        just_final = rmse(total_loss, num_samples)
        return error, just_final

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

def rmse(total_loss, num_samples):
    return np.sqrt(total_loss / num_samples)

# for local testing
#torch.set_num_threads(48)

if __name__ == '__main__':
    # gpu stuff
    print(f"Running on {device}")
    # basic initialization
    model = LSTMModel()
    model = model.to(device, non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # taken from the paper
    loss_fn = nn.MSELoss() # taken from the paper
    subjects = [f for f in os.listdir(os.getcwd()) if re.search("AB\d+", f)]
    test_subjects = ['AB01']
    training_subjects = [s for s in subjects if s not in test_subjects]
    activities = re.compile(".");
    #activities = re.compile("normal_walk"); # smaller dataset
    print(f"initializing training dataset... {curtime()}")
    training_data = dataloader.GreedyGrandLSTMDataset(training_subjects, activities)
    print(f"done initializing training dataset... {curtime()}")
    print(f"initializing test dataset... {curtime()}")
    test_data = dataloader.GreedyGrandLSTMDataset(test_subjects, activities)
    print(f"done initializing test dataset... {curtime()}")
    # I'm pretty sure prefetching is useless if we're doing CPU training
    # unless the disk IO is really slow, but I'm hoping for gpu we can make
    # better use of both resources
    train_dataloader = torch.utils.data.DataLoader(
            training_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=(device == "cuda"),
            #num_workers=(2 if device == "cuda" else 0),
            #persistent_workers=(device == "cuda"),
            )
    test_dataloader = torch.utils.data.DataLoader(
            test_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=(device == "cuda"),
            #num_workers=(2 if device == "cuda" else 0),
            #persistent_workers=(device == "cuda"),
            )

    if os.path.isfile(checkpoint_path) and len(sys.argv) > 1:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}... {curtime()}")
    else:
        start_epoch = 0
        print(f"No checkpoint found since {os.path.isfile(checkpoint_path)} and {len(sys.argv) > 1}. Starting training from scratch.... {curtime()}")
    should_early_stop = EarlyStop() # taken from the paper
    last_save_time = time.time()
    last_eval_time = 0
    for epoch in range(start_epoch, n_epochs):
        #print(f"epoch {epoch} at {curtime()}")
        model.train()
        total_training_loss = 0.0 # sum of losses of all batches
        num_samples = 0 # number of total samples trained on
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            y_pred = model(X_batch) # happens on gpu
            # TODO:
            # I don't know pytorch/python/gpus enough, but this assert may
            # cause the tensor to shuttle back to the cpu memory
            assert not y_pred.isnan().any()
            # take the loss wrt the true moments at each timestamp in the window
            loss = loss_fn(y_pred, y_batch)
            assert not loss.isnan().any()
            optimizer.zero_grad() # this can go anywhere except for between backward and step
            loss.backward()
            optimizer.step()
            # on cpu, but should be fast
            batch_size = X_batch.size()[0]
            total_training_loss += loss.item() * batch_size
            num_samples += batch_size
            #print(f"after batch {curtime()}")
        # save checkpoint
        #print(f"saving model at {curtime()}")
        if time.time() > last_save_time + 1800:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'loss': loss
                }, checkpoint_path)
            last_save_time = time.time()
        if time.time() > last_eval_time + 1800: # only eval every n epochs
        #if epoch % 10 == 0: # only eval every n epochs
            model.eval()
            with torch.no_grad():
                train_rmse, train_rmse_just_final = eval_rmse(train_dataloader)
                test_rmse, test_rmse_just_final = eval_rmse(test_dataloader)
                print("Epoch %d: whole window: train RMSE %.4f, test RMSE %.4f", epoch, train_rmse, test_rmse_just_final)
                print("Epoch %d: final timestamp: train RMSE %.4f, test RMSE %.4f", epoch, train_rmse, test_rmse_just_final)
            last_eval_time = time.time()
        if should_early_stop.should_early_stop(total_training_loss):
            print(f"Stopping early on epoch {epoch}, with training RMSE %.4f ... {curtime()}", rmse(total_training_loss, num_samples))
            break
    print(f"Finished Training at {curtime()}")
