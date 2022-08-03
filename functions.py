# Kim van Hoeve
# 3/8/2022
# General Functions 


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
    """Checks if a file is unzipped and unzips it in target """
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


# =============== loop implementation ==============
# train and test loop https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html


def train_loop(dataloader, model, loss_fn, optimizer, epoch_index, device):
    """Loops through training data and makes predictions for the data.
    Records the loss claculated by the loss functions."""
    # import pacakges
    import torch
    import torch.cuda
    
    # turn on model train mode
    model.train()

    # assigning important variables
    running_loss = 0.
    last_loss = 0.
    size = len(dataloader.dataset)

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

        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            last_loss = running_loss/100
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.

    return last_loss


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
            for i in range(len(y)):
                for num in torch.round(y[i], decimals=1) == torch.round(pred[i], decimals=1):
                    if num:
                        correct += 1

    eval_loss /= num_batches
    eval_epoch_acc = correct / (size * 12)
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

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            testloop_loss += loss_fn(pred, y).item()
            for i in range(len(y)):
                for num in torch.round(y[i], decimals=1) == torch.round(pred[i], decimals=1):
                    if num:
                        correct += 1

    testloop_loss /= num_batches
    test_acc = correct / (size * 12)
    print(f"Test Error: \n Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {testloop_loss:>8f} \n")
    return testloop_loss, test_acc

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
    matplotlib.use('TkAgg')
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


# =================================================================================================================
