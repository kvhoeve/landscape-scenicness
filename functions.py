# kvhoeve
# 3/8/2022
# General Functions
# GNU General Public License v3.0


# Taken from the dataloader PyTorch tutorial at 4/5/2022
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

# Importing needed libraries
from __future__ import print_function, division


# =============== make directories ==============

def make_dir(rel_path, folder=""):
    """ Checks if a directory with the relative path given exists.
    If it doesn't it makes the directory."""
    # import packages
    import os
    import errno
    
    try:
        os.mkdir(os.path.join(rel_path, folder))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.join(rel_path, folder)):
            pass


# =============== loading data ==============

def unzip_file(root_path, file_name=""):
    """Checks if a file is unzipped and unzips it in target path."""
    # import packages
    import os
    import zipfile
    
    # unzip data files
    if os.path.isdir(os.path.join(root_path, file_name)):
        pass
    else:
        print("Unzipping data file")
        with zipfile.ZipFile(os.path.join(root_path, (file_name + '.zip')), "r") as zip_ref:
            zip_ref.extractall(root_path)


def extract_tarfile(root_path, file_name="", doc_name=""):
    """Checks if a file is unzipped and unzips it in target """
    # import packages
    import os
    import tarfile
    
    # unzip data files
    if os.path.isdir(os.path.join(root_path, doc_name)):
        pass
    else:
        print("Extracting data from tarfile")
        with tarfile.TarFile(os.path.join(root_path, (file_name + '.tar')), "r") as tar_ref:
            tar_ref.extractall(root_path)


# ============== learning rate decay ========

def exp_decay(epoch, initial_lrate=0.001, k=0.1):
    # made with inspiration from Suki Lau on 7/8/2022 on:
    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    """Decreases the learning rate with the number of epochs. 
    Needs an initial learning rate and a factor (k) to decrease the learning rate with."""
    # import package
    import math
       
    lrate = initial_lrate * math.exp(-k*epoch)
       
    return lrate


# =============== loop implementation ==============
# train and test loop https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html


def train_loop(dataloader, model, loss_fn, optimizer, epoch_index, device):
    """Loops through training data and makes predictions for the data.
    Records the loss claculated by the loss functions."""
    # import pacakges
    import torch
    import torch.cuda
    import numpy as np
    import pandas as pd
    
    # turn on model train mode
    model.train()

    # assigning important variables
    running_loss = 0.
    last_loss = 0.
    size = len(dataloader.dataset)
    target_list = []
    prediction_list = []
    optimizer.param_groups[0]["lr"] = exp_decay(epoch=epoch_index)

    for batch, (X, y) in enumerate(dataloader):
        # place data on device
        X = X.to(device)
        y = y.to(device).float()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # append labels to arrays for plotting
        for t in range(len(y)):
            for i in range(len(y[1])):
                target_label = float(torch.round(y[t][i], decimals=1).detach().cpu())
                prediction_label = float(torch.round(pred[t][i], decimals=1).detach().cpu())
                target_list.append(target_label)
                prediction_list.append(prediction_label)
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            last_loss = running_loss/100
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.

    
    return last_loss, target_list, prediction_list


def eval_loop(dataloader, model, loss_fn, device):
    """Ëvaluation of the model on the evaluation dataset. Used for tuning."""
    # import packages
    import torch
    import torch.cuda
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    eval_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            # keeping track of the number of correctly predicted numbers
            for t in range(len(y)):
                for num in torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
                    if num:
                        correct += 1

    eval_loss /= num_batches
    eval_epoch_acc = correct / (size * len(y[1]))
    print(f"Eval Error: \n Accuracy: {(100*eval_epoch_acc):>0.1f}%, Avg loss: {eval_loss:>8f} \n")
    return eval_loss, eval_epoch_acc


def test_loop(dataloader, model, loss_fn, device):
    # import packages
    import torch
    import torch.cuda
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    testloop_loss, correct = 0, 0
    target_scorelist = []
    prediction_scorelist = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            testloop_loss += loss_fn(pred, y).item()
            
            # tally correct values for accuracy
            for i in range(len(y)):
                for num in torch.round(y[i], decimals=1) == torch.round(pred[i], decimals=1):
                    if num:
                        correct += 1

                target_score = float(torch.round(y[i][1], decimals=1).detach().cpu())
                prediction_score = float(torch.round(pred[i][1], decimals=1).detach().cpu())
                target_scorelist.append(target_score)
                prediction_scorelist.append(prediction_score)
    
    # metrics                    
    testloop_loss /= num_batches
    test_acc = correct / (size * len(y[1]))
    s_r = spearmanr(torch.Tensor(target_scorelist), torch.Tensor(prediction_scorelist))
    print(f"Test Error: \n Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {testloop_loss:>8f} \n")
    print(f"Spearman correlation on the aesthetic score: {s_r:.2f}")
    return testloop_loss, test_acc, s_r

# =================== performance functions ====================

def performance_overview(e_list, 
                         t_list, 
                         v_list, 
                         a_list,                           
                         file_name,
                         file_path='./data/AADB/models'):
    """Creates a pandas dataframe with perfromance statistics for an overview.
    Returns the pandas dataframe, but saves it as a text file."""    
    # Import packages
    import os
    import pandas as pd

    # create a csv file with the accuracies over time
    acc_dict = {"epochs": e_list,
                "training loss": t_list,
                "validation loss": v_list,
                "validation accuracy" : a_list}
    run_data = pd.DataFrame(acc_dict)
    run_data.to_csv(os.path.join(file_path, file_name), sep=" ")
    return(run_data)

def performance_fig(epochs_list, training_loss, validation_loss, fig_name, fig_path="./data/AADB/figures"):
    # taken from the pyplot manual on 26/7/2022
    # https://matplotlib.org/stable/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    """Creates a figure showing the performance of the model. Uses matplotlib pyplot """
    #import packages
    import os
    import matplotlib
   # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()   # interactive mode    
        
    fig, ax = plt.subplots(figsize=(15, 7.5), layout='constrained')
    ax.plot(epochs_list, training_loss, label='Training loss')
    ax.plot(epochs_list, validation_loss, label= 'Validation loss')
    ax.set_title('Train loss and Validation loss over epochs AADB')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.set_xticks(range(1, len(epochs_list) + 1, 1))
    ax.legend()
    plt.savefig(os.path.join(fig_path, fig_name))

    return(ax)
    
def prediction_fig(target, prediction):
    # taken from the pyplot manual on 26/7/2022
    # https://matplotlib.org/stable/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    """Creates a figure showing the performance of the model. Uses matplotlib pyplot """
    #import packages
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()   # interactive mode    
    
    # create scatterplot
    plt.scatter(target, prediction)    
    
    z = np.polyfit(target, prediction, 1)
    p = np.poly1d(z)
    plt.plot(target,p(target),"r--")
    plt.xlabel("target labels (y)")
    plt.ylabel("prediction labels (y^)")
    
    return plt.show()

def spearmanr(x1, x2):
    """Takes two torch tensors and first calculates the covaraince (num) and
    the producct of the sqaure root to divide them and calculate the Pearson correlation. """
    # import package
    import torch
    
    vx = x1 - torch.mean(x1)
    vy = x2 - torch.mean(x2)

    num = torch.sum(vx * vy)
    den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))

    return num / den
    
# =================================================================================================================
