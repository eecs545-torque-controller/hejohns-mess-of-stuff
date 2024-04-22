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
import datetime
import pytz
from torch.profiler import profile, record_function, ProfilerActivity
# our files
import config
import load

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2):
        super().__init__()
        self.input_size = len(config.sensor_list)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
                input_size=self.input_size, # number of expected features in input
                # TODO: trial and error
                hidden_size=hidden_size, # number of features in hidden state
                # TODO: trial and error
                num_layers=num_layers, # number of recurrent layers
                batch_first=True,
                )
        self.linear = nn.Linear(hidden_size, len(config.output_list))
        #self.dropout = nn.Dropout(p=0.4) # paper used p=0.4
    def forward(self, x):
        #print("33")
        #print(x.dtype)
        x, _ = self.lstm(x)
        #assert x.size() == (<config.batch_size, config.window_size, self.hidden_size)
        #x = x[:, -1, :]
        x = self.linear(x)
        ## we're only interested in the moments at the last timestamp
        #x = x[:, -1, :]
        return x

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

checkpoint_path = 'saved_model.ckpt'
def curtime():
    return datetime.datetime.now(tz=pytz.timezone('US/Eastern')).time()
torch.set_num_threads(48)

if __name__ == '__main__':
    # basic initialization
    model = LSTMModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # taken from the paper
    loss_fn = nn.MSELoss() # taken from the paper
    #subjects = [f for f in os.listdir('.') if re.search("AB\d+", f)]
    subjects = ['AB02', 'AB03']
    #activities = re.compile(".");
    activities = re.compile(".");
    print(f"initializing training dataset... {curtime()}")
    training_data = load.GreedyLSTMDataset(subjects, activities)
    print(f"initializing test dataset... {curtime()}")
    test_data = load.GreedyLSTMDataset(['AB01'], activities)
    # I'm pretty sure prefetching is useless if we're doing CPU training
    # I'm just using num_workers=2 so we can set persistent_workers=True
    train_dataloader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=config.batch_size, num_workers=2, persistent_workers=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=config.batch_size, num_workers=2, persistent_workers=True)

    if os.path.isfile(checkpoint_path) and len(sys.argv) > 1:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")
    should_early_stop = EarlyStop() # taken from the paper
    n_epochs = 4000 # taken from the paper
    for epoch in range(start_epoch, n_epochs):
        #print(f"epoch {epoch} at {curtime()}")
        model.train()
        #with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        #    with record_function("training_loop"):
        sloss = 0.0
        snb = 0
        for X_batch, y_batch in train_dataloader:
            #print(f"before batch {curtime()}")
            #with record_function("forward"):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            sloss += loss.item() * X_batch.size()[0]
            snb += 1
            optimizer.zero_grad()
            #with record_function("backward"):
            loss.backward()
            optimizer.step()
                    #print(f"after batch {curtime()}")
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        # save checkpoint
        #print(f"saving model at {curtime()}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss
            }, checkpoint_path)
        if epoch % 100 == 0: # only eval every n epochs
            model.eval()
            with torch.no_grad():
                total_training_loss = 0.0
                num_training_batches = 0
                for X_batch, y_batch in train_dataloader:
                    y_pred = model(X_batch)
                    batch_loss = loss_fn(y_pred, y_batch)
                    total_training_loss += batch_loss.item()
                    num_training_batches += 1
                total_test_loss = 0.0
                num_test_batches = 0
                for X_batch, y_batch in test_dataloader:
                    y_pred = model(X_batch)
                    batch_loss = loss_fn(y_pred, y_batch)
                    total_test_loss += batch_loss.item()
                    num_test_batches += 1
                #train_rmse = np.sqrt(loss_fn(y_pred, y_train))
                #y_pred = model(X_test)
                #test_rmse = np.sqrt(loss_fn(y_pred, y_test))
                train_rmse = np.sqrt(total_training_loss / num_training_batches)
                test_rmse = np.sqrt(total_test_loss / num_test_batches)
                print("Epoch %d: entire: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
                total_training_loss = 0.0
                num_training_batches = 0
                for X_batch, y_batch in train_dataloader:
                    y_pred = model(X_batch)
                    y_pred = y_pred[:, -1, :]
                    y_batch = y_batch[:, -1, :]
                    batch_loss = loss_fn(y_pred, y_batch)
                    total_training_loss += batch_loss.item()
                    num_training_batches += 1
                total_test_loss = 0.0
                num_test_batches = 0
                for X_batch, y_batch in test_dataloader:
                    y_pred = model(X_batch)
                    y_pred = y_pred[:, -1, :]
                    y_batch = y_batch[:, -1, :]
                    batch_loss = loss_fn(y_pred, y_batch)
                    total_test_loss += batch_loss.item()
                    num_test_batches += 1
                #train_rmse = np.sqrt(loss_fn(y_pred, y_train))
                #y_pred = model(X_test)
                #test_rmse = np.sqrt(loss_fn(y_pred, y_test))
                train_rmse = np.sqrt(total_training_loss / num_training_batches)
                test_rmse = np.sqrt(total_test_loss / num_test_batches)
                print("Epoch %d: last: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        if should_early_stop.should_early_stop(sloss):
            print(f"Stopping early on epoch {epoch}, with training RMSE %.4f", np.sqrt(sloss / snb))
            break
