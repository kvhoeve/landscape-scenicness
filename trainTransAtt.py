# Kim van Hoeve
# 2/8/2022
# transient attributes model

# Importing needed libraries
from __future__ import print_function, division
from datetime import datetime, timedelta
import torch
import torch.cuda
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ============== functions ==================
# Import classes and functions specialized for this dataset
from classesTransAtt import *
from functions import *


# =============== Important variables ==============
# for reprodicibility
torch.manual_seed(213)

# variables for paths
base_path = "./data/transient_attributes"
run_path = "runs"
model_path = "models"
fig_path = "figures"
cam_path = "cam"


# =============== make directories ==============
# necessary directories for training and saving the model
make_dir(base_path)
make_dir(base_path, run_path)
make_dir(base_path, model_path)
make_dir(base_path, fig_path)
make_dir(base_path, cam_path)

# ================ Access folders ==================
# If you haven't, please download the necessary files:
# Images: http://transattr.cs.brown.edu/files/aligned_images.tar (1.8 GB)
# Labels: http://transattr.cs.brown.edu/files/annotations.tar (3.6 MB)

extract_tarfile(base_path, "annotations", doc_name="annotations")
extract_tarfile(base_path, "unaligned_images", doc_name="imageLD")
extract_tarfile(base_path, "training_test_splits", doc_name="holdout_split")

# =============== transform ==============
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =============== hyper parameters ==============

epochs = 50
b_size = 8
num_worker = 0
early_stop_tol = 5
early_stop_epoch = 1
epoch_num = 0
best_eval_loss = 1_000_000.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'resnet50'


# =============== model ==============
# Loading in the ResNet-X model from the Pytorch repository
# model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Sequential(
#    nn.Linear(num_features, 40),
#    nn.Sigmoid())

model = TACAM()
model.to(device)

# =============== Dataloaders ==============
# split the data into train (80%), validation (10%) and test (10%) sets

train_data, val_data = random_split(dataset=TransAttributes(root=base_path, split="train", transform=preprocess), lengths=[6047, 857])
test_data = TransAttributes(root=base_path, split="test", transform=preprocess)

train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)
eval_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True, num_workers=num_worker, pin_memory=True)


# =============== loss & optimizer ==============
    
loss_fn = nn.MSELoss()
# learning rate is scheduled in the training loop (see functions.py)
optimizer = torch.optim.Adam(model.parameters())


# =============== calling ==============
# setting time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('./data/transient_attributes/runs/TA_{}_trainer_{}'.format(model_name, timestamp))


train_loss_list = []
eval_loss_list = []
eval_acc_list = []
epoch_list = list(range(1, epochs + 1, 1))
early_stop = 0

# record time passing
start_time = datetime.now().strftime('%X')
d1 = timedelta(hours=int(start_time[0:2]), minutes=int(start_time[3:5]), seconds=int(start_time[6:8]))


for t in range(epochs):
    print(f"Epoch {epoch_num + 1}\n-------------------------------")
    avg_train_loss, lbl_list, pred_list = cam_train(train_dataloader, model, loss_fn, optimizer, epoch_index=t, device=device)
    train_loss_list.append(avg_train_loss)
    # visualize training
    prediction_fig(lbl_list, pred_list)
    
    # evaluation
    avg_eval_loss, eval_acc = cam_eval(eval_dataloader, model, loss_fn, device=device)
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
        model_path = './data/transient_attributes/models/model_{}_{}_{}.pth'.format(model_name, timestamp, epoch_num + 1)
        torch.save(model, model_path)
        early_stop = 0
    else:
        # implement early stopping if tolerance is crossed
        early_stop += 1
        if early_stop >= early_stop_tol:
            # performance overview
            performance_overview(list(range(1, early_stop_epoch + 1, 1)), train_loss_list, eval_loss_list, eval_acc_list, file_path='./data/transient_attributes/models', file_name='model_{}_overview_{}.txt'.format(model_name, timestamp))
            # performance figure
            performance_fig(list(range(1, early_stop_epoch + 1, 1)), train_loss_list, eval_loss_list, fig_path='./data/transient_attributes/figures', fig_name='model_{}_overview_{}.png'.format(model_name, timestamp))
            print("We are stopping at epoch:", epoch_num + 1)
            break

    epoch_num += 1
    early_stop_epoch += 1

print("Finished training!")


# record time passing
stop_time = datetime.now().strftime('%X')
d2 = timedelta(hours=int(stop_time[0:2]), minutes=int(stop_time[3:5]), seconds=int(stop_time[6:8]))
time_pass = d2 - d1
print("The time passed: " + str(time_pass))


# create a performance overview
performance_overview(epoch_list, train_loss_list, eval_loss_list, eval_acc_list, file_path='./data/transient_attributes/models', file_name='model_{}_overview_{}.txt'.format(model_name, timestamp))

# create performance figure
performance_fig(epoch_list, train_loss_list, eval_loss_list, fig_path='./data/transient_attributes/figures', fig_name='model_{}_overview_{}.png'.format(model_name, timestamp))

# =================================================================================================================
