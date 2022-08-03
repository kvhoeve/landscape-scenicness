# Kim van Hoeve
# 4/5/2022
# Training AADB model
# GNU Public Lisence v3.0


# Taken from the dataloader PyTorch tutorial at 4/5/2022
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

# Importing needed libraries
from __future__ import print_function, division
from datetime import datetime, timedelta
import torch
import torch.cuda
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# ============== functions ==================
# Import classes and functions specialized for this dataset
from classesAADB import *
from functions import *


# =============== Important variables ==============
# variables for paths
base_path = "./data/AADB"
run_path = "runs"
model_path = "models"
fig_path = "figures"



# =============== make directories ==============

make_dir(base_path)
make_dir(base_path, run_path)
make_dir(base_path, model_path)
make_dir(base_path, fig_path)


# ================ Unzip folders ==================
# If you haven't, please download the necessary files:
# Images: https://drive.google.com/uc?export=download&id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_ (2 GB)
# Labels: (download imgListFiles_label.zip) (2 MB)
# https://drive.google.com/drive/folders/0BxeylfSgpk1MOVduWGxyVlJFUHM?resourcekey=0-qecf-sZVexPbF6XLU4Gq_g&usp=sharing

unzip_file(base_path, file_name="datasetImages_originalSize")
unzip_file(base_path, file_name="imgListFiles_label")


# =============== transform ==============
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =============== hyper parameters ==============

epochs = 1
b_size = 24
learning_rate = 1e-3
num_worker = 0
early_stop_tol = 5
epoch_num = 0
best_eval_loss = 1_000_000.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'resnet18'


# =============== model ==============
# Loading in the ResNet-X model from the Pytorch repository

model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 12),
    nn.Tanh())
model.to(device)


# =============== Dataloaders ==============

train_data = AADB(split='train', root=base_path, transform=preprocess)
test_data = AADB(split='test', root=base_path, transform=preprocess)
val_data = AADB(split='val', root=base_path, transform=preprocess)

train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)
eval_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)



# =============== loss & optimizer ==============

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# =============== calling ==============
# setting time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('./data/AADB/runs/AADB_{}_trainer_{}'.format(model_name, timestamp))


train_loss_list = []
eval_loss_list = []
eval_acc_list = []
epoch_list = list(range(1, epochs + 1, 1))

# record time passing
start_time = datetime.now().strftime('%X')
d1 = timedelta(hours=int(start_time[0:2]), minutes=int(start_time[3:5]), seconds=int(start_time[6:8]))


for t in range(epochs):
    print(f"Epoch {epoch_num + 1}\n-------------------------------")
    avg_train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch_index=epochs, device=device)
    train_loss_list.append(avg_train_loss)
    avg_eval_loss, eval_acc = eval_loop(eval_dataloader, model, loss_fn, device=device)
    eval_loss_list.append(avg_eval_loss)
    eval_acc_list.append(eval_acc)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_train_loss, 'Validation' : avg_eval_loss },
                    epoch_num + 1)
    writer.flush()

    # Track the best performance, and save the model's state
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        model_path = './data/AADB/models/model_{}_{}_{}.pth'.format(model_name, timestamp, epoch_num + 1)
        torch.save(model, model_path)
    else:
        # implement early stopping if tolerance is crossed
        early_stop += 1
        if early_stop >= early_stop_tol:
            print("We are stopping at epoch:", epoch_num + 1)
            break

    epoch_num += 1

print("Finished training!")


# record time passing
stop_time = datetime.now().strftime('%X')
d2 = timedelta(hours=int(stop_time[0:2]), minutes=int(stop_time[3:5]), seconds=int(stop_time[6:8]))
time_pass = d2 - d1
print("The time passed: " + str(time_pass))


# create a performance overview
performance_overview(epoch_list, train_loss_list, eval_loss_list, eval_acc_list, file_name='model_{}_overview_{}.txt'.format(model_name, timestamp))

# create performance figure
performance_fig(epoch_list, train_loss_list, eval_loss_list, fig_name='model_{}_overview_{}.png'.format(model_name, timestamp))

# =================================================================================================================
