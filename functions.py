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
    
    # turn on model train mode
    model.train()

    # assigning important variables
    running_loss = 0.
    last_loss = 0.
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    target_list = []
    prediction_list = []
    optimizer.param_groups[0]["lr"] = exp_decay(epoch=epoch_index)

    for batch, (X, y, p) in enumerate(dataloader):
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
                target_label = float(y[t][i].detach().to('cpu'))
                prediction_label = float(pred[t][i].detach().to('cpu'))
                target_list.append(target_label)
                prediction_list.append(prediction_label)
        
      #  for t in range(len(y)):        
      #      target_label = float(y[t].detach().to('cpu'))
      #      prediction_label = float(pred[t].detach().to('cpu'))
      #      target_list.append(target_label)
      #      prediction_list.append(prediction_label)
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

    last_loss = running_loss/num_batches    

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
        for X, y, p in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            # keeping track of the number of correctly predicted numbers
           
          #  for t in range(len(y)):
          #      if torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
          #          correct += 1
            for t in range(len(y)):
               for num in torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
                   if num:
                       correct += 1

    eval_loss /= num_batches
    eval_epoch_acc = correct / (size * len(y[1]))
    #eval_epoch_acc = correct / size
    print(f"Eval Error: \n Accuracy: {(100*eval_epoch_acc):>0.1f}%, Avg loss: {eval_loss:>8f} \n")
    
    return eval_loss, eval_epoch_acc


def test_loop(dataloader, model, loss_fn, device, attributes=None):
    # import packages
    import torch
    import torch.cuda
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    testloop_loss, correct = 0, 0
    target_list = []
    prediction_list = []
    spearman_list = []

    with torch.no_grad():
        for X, y, p in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            testloop_loss += loss_fn(pred, y).item()
            
            for t in range(len(y)):
                target_label = float(y[t].detach().to('cpu'))
                prediction_label = float(pred[t].detach().to('cpu'))
                target_list.append(target_label)
                prediction_list.append(prediction_label)
                
                
                if torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
                    correct += 1
                    
            
            # tally correct values for accuracy
          #  for t in range(len(y)):
          #      for num in torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
          #          if num:
          #              correct += 1
                        
                # create long lists with target and predictions for all images in the set
              #  for i in range(len(y[1])):
              #      target_label = float(y[t][i].detach().to('cpu'))
              #      prediction_label = float(pred[t][i].detach().to('cpu'))
              #      target_list.append(target_label)
              #      prediction_list.append(prediction_label)
        
                            
    
                    
    # metrics                    
    testloop_loss /= num_batches
   # test_acc = correct / (size * len(y[1]))
    test_acc = correct / size
    
    # calculate the 
   # for u in range(len(y[0])):
   #     att_targetscore = []
   #     att_predscore = []
   #     for att in range(u, len(target_list), len(y[1])):
   #         att_targetscore.append(target_list[att])
   #         att_predscore.append(prediction_list[att])
        
    #    s_r = spearmanr(torch.Tensor(att_targetscore), torch.Tensor(att_predscore))
    s_r = spearmanr(torch.Tensor(target_list), torch.Tensor(prediction_list))
    spearman_list.append(s_r)
        
    # printing results    
    print(f"Test Error: \n Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {testloop_loss:>8f} \n")
    
    # printing every spearman correlation value for every attribute
    for s in range(len(spearman_list)):
        if attributes == None:
            print(f"Spearman correlation on attribute {s} : {spearman_list[s]:>.2f}")
        else:
            print(f"Spearman correlation on attribute {attributes[s]} : {spearman_list[s]:>.2f}")
    
    # create a prediction figure
    prediction_fig(target_list, prediction_list)
        
    #return testloop_loss, test_acc, target_list, prediction_list

# adapted from isaaccorley on 30/08/2022 at
# https://github.com/isaaccorley/deep-aesthetics-pytorch/blob/main/train.py
def aadb_train(dataloader, model, loss_fn, optimizer, epoch_index, device):
    """Loops through training data and makes predictions for the data.
    Records the loss claculated by the loss functions."""
    # import pacakges
    import torch
    import torch.cuda
    
    # turn on model train mode
    model.train()

    # assigning important variables
    running_loss = 0.
    running_reg = 0.
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    optimizer.param_groups[0]["lr"] = exp_decay(epoch=epoch_index)

    for batch, (X, y, p) in enumerate(dataloader):
        # place data on device
        X = X.to(device)
        y = y.to(device).float()
        # split batches in two to be able to rank them
        x1, x2 = torch.split(X, int(len(X)/2), dim=0)
        y1, y2 = torch.split(y, int(len(X)/2), dim=0)

        # Compute prediction and loss
        pred1 = model(x1)
        pred2 = model(x2)
        loss, loss_reg, loss_rank = loss_fn(y_pred=(pred1, pred2), 
                                            y_true=(y1, y2))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_reg += loss_reg.item()
        if batch % 100 == 0:
           loss, current = loss.item(), batch * len(X)
           print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
           
          
    last_loss = running_loss/num_batches
    last_reg =  running_reg/num_batches      
    print("Epoch {}, Combined Loss: {:.4f}, Reg Loss: {:.4f}, Rank Loss: {:.4f} ".format(
        epoch_index+1, float(loss), float(loss_reg), float(loss_rank)))

    
    return last_loss, last_reg

def tuned_train(dataloader, model, loss_fn, optimizer, epoch_index, device):
    """Loops through training data and makes predictions for the data.
    Records the loss claculated by the loss functions. This loop has been altered
    to boost the losses from """
    # import pacakges
    import torch
    import torch.cuda
    
    # turn on model train mode
    model.train()

    # assigning important variables
    running_loss = 0.
    last_loss = 0.
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    target_list = []
    prediction_list = []
    optimizer.param_groups[0]["lr"] = exp_decay(epoch=epoch_index)

    for batch, (X, y, p) in enumerate(dataloader):
        # place data on device
        X = X.to(device)
        y = y.to(device).float()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        w_loss = torch.cat(tensors=(loss[:,0], loss[:,1]*2 , loss[:,2], 
                                    loss[:,3], loss[:,4], loss[:,5], loss[:,6]*5,
                                    loss[:,7], loss[:,8]*2, loss[:,9]*2,
                                    loss[:,10]*5, loss[:,11]), dim=0)
        weighted_loss = torch.mean(w_loss)
        

        # Backpropagation
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        # append labels to arrays for plotting
        for t in range(len(y)):
            for i in range(len(y[1])):
                target_label = float(y[t][i].detach().to('cpu'))
                prediction_label = float(pred[t][i].detach().to('cpu'))
                target_list.append(target_label)
                prediction_list.append(prediction_label)
        
        running_loss += weighted_loss.item()
        if batch % 100 == 0:
            l, current = weighted_loss.item(), batch * len(X)
            print(f"loss: {l:>7f}  [{current:>5d}/{size:>5d}]")
            
    last_loss = running_loss/num_batches
    
    return last_loss, target_list, prediction_list

def tuned_eval(dataloader, model, loss_fn, device):
    """Ëvaluation of the model on the evaluation dataset. Used for tuning."""
    # import packages
    import torch
    import torch.cuda
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    eval_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, p in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            pred = model(X)
            loss = loss_fn(pred, y)
            w_loss = torch.cat(tensors=(loss[:,0], loss[:,1]*2 , loss[:,2], 
                                        loss[:,3], loss[:,4], loss[:,5], loss[:,6]*5,
                                        loss[:,7], loss[:,8]*2, loss[:,9]*2,
                                        loss[:,10]*5, loss[:,11]), dim=0)
            weighted_loss = torch.mean(w_loss)
            
            eval_loss += weighted_loss.item()
            # keeping track of the number of correctly predicted numbers
            for t in range(len(y)):
                for num in torch.round(y[t], decimals=1) == torch.round(pred[t], decimals=1):
                    if num:
                        correct += 1

    eval_loss /= num_batches
    eval_epoch_acc = correct / (size * len(y[1]))
    print(f"Eval Error: \n Accuracy: {(100*eval_epoch_acc):>0.1f}%, Avg loss: {eval_loss:>8f} \n")
    
    return eval_loss, eval_epoch_acc
# =================== performance functions ====================

def performance_overview(e_list, 
                         t_list,
                         v_list, 
                         a_list,                           
                         file_name,
                         r_list=None,
                         file_path='./data/AADB/models'):
    """Creates a pandas dataframe with perfromance statistics for an overview.
    Returns the pandas dataframe, but saves it as a text file."""    
    # Import packages
    import os
    import pandas as pd

    # create a csv file with the accuracies over time
    if r_list == None:
        acc_dict = {"epochs": e_list,
                    "training loss": t_list,
                    "validation loss": v_list,
                    "validation accuracy" : a_list}
    else:
        acc_dict = {"epochs": e_list,
                    "training loss": t_list,
                    "regression loss": r_list,
                    "validation loss": v_list,
                    "validation accuracy" : a_list}
        
    run_data = pd.DataFrame(acc_dict)
    run_data.to_csv(os.path.join(file_path, file_name), sep=" ")
    return(run_data)

def performance_fig(epochs_list, training_loss, validation_loss, fig_name, extra_list=None, fig_path="./data/AADB/figures"):
    # taken from the pyplot manual on 26/7/2022
    # https://matplotlib.org/stable/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
    """Creates a figure showing the performance of the model. Uses matplotlib pyplot """
    #import packages
    import os
   # import matplotlib
   # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()   # interactive mode    
        
    fig, ax = plt.subplots(figsize=(25, 12.5))
    ax.plot(epochs_list, training_loss, label='Training loss')
    ax.plot(epochs_list, validation_loss, label= 'Validation loss')
    
    if extra_list != None:
        ax.plot(epochs_list, extra_list, label='Regression loss')
    else:
        pass
    
    ax.set_title('Train loss and Validation loss over epochs')
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
    
# ======================= CAM implementation ========================================


def cam_train(dataloader, model, loss_fn, optimizer, epoch_index, device):
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
    num_batches = len(dataloader)
    target_list = []
    prediction_list = []
    optimizer.param_groups[0]["lr"] = exp_decay(epoch=epoch_index)

    for batch, (X, y, p) in enumerate(dataloader):
        # place data on device
        X = X.to(device)
        y = y.to(device).float()

        # Compute prediction and loss
        cam, pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # append labels to arrays for plotting
        for t in range(len(y)):
            for i in range(len(y[1])):
                target_label = float(y[t][i].detach().to('cpu'))
                prediction_label = float(pred[t][i].detach().to('cpu'))
                target_list.append(target_label)
                prediction_list.append(prediction_label)
        
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    last_loss = running_loss/num_batches
    
    return last_loss, target_list, prediction_list


def cam_eval(dataloader, model, loss_fn, device):
    """Ëvaluation of the model on the evaluation dataset. Used for tuning."""
    # import packages
    import torch
    import torch.cuda
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    eval_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, p in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            cam, pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            
            # flattening tensor from (batch_size, att, 1, 1) to (batch_size, att)
            flat_pred1 = pred.squeeze(-1)
            flat_pred2 = flat_pred1.squeeze(-1)
            # keeping track of the number of correctly predicted numbers
            for t in range(len(y)):
                    for num in torch.round(y[t], decimals=1) == torch.round(flat_pred2[t], decimals=1):
                        if num:
                            correct += 1

    eval_loss /= num_batches
    eval_epoch_acc = correct / (size * len(y[0]))
    print(f"Eval Error: \n Accuracy: {(100*eval_epoch_acc):>0.1f}%, Avg loss: {eval_loss:>8f} \n")
    
    return eval_loss, eval_epoch_acc


def CAM_loop(dataloader, model, save_path, device, attribute_list, threshold = 0.8, dataset = 'TA'):
    """Dataset argument shows which dataset is used to enuse correct save path. Saves an image with the input image
    and the resulting Class Activation Map in a file of your choice (save_path).
    One heatmap generated per attribute.
    Model can be a path or ResNet object. cv2 image visualization was partly
    taken from zhoubolei on 17/8/22 on:
        https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py"""
    # import packages
    import torch
    import torch.cuda
    import cv2
    import numpy as np
    import os

    
    # initialize the model
    model.to(device)
    
    # loop though the data
    with torch.no_grad():
        for X, y, p in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            cam, pred = model(X)
            
    
            for t in range(len(cam)):
                for num in range(len(pred)):
                    if pred[t][num].item() >= threshold:
                        img = cv2.imread(p[t])
                        np_cam = np.array(cam[t].reshape(len(attribute_list),7,7).detach().to('cpu')*255, dtype=np.uint8)
                        height, width, _ = img.shape
                
                        heatmap = cv2.applyColorMap(cv2.resize(np_cam[num],(width, height)), cv2.COLORMAP_JET)
                        result = cv2.addWeighted(heatmap, 0.3, img, 0.7, 0)
                        
                        if dataset == 'TA':
                            split_1 = p[t].split('\\')
                            split_2 = split_1[-1]
                            split_3 = split_2[:-5].split('/')
                            save_name = split_3[0] + '_' + split_3[1] + '_' + attribute_list[num] + '_' + str(round(float(pred[t][num].item()), 2)) + '.png'
                            cv2.imwrite(os.path.join(save_path, save_name), result)
                        
                        elif dataset == 'AADB':
                            split_1 = p[t].split('\\')
                            split_2 = split_1[-1].split('.')
                            save_name = split_2[0] + '_' + attribute_list[num] + '_' + str(round(float(pred[t][num]), 2)) + '.png'
                            cv2.imwrite(os.path.join(save_path, save_name), result)
                        
                        else:
                            print('Does nothing')