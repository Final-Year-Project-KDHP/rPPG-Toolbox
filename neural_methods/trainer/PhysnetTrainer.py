"""PhysNet Trainer."""
print
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Subset


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")



    def train(self, data_loader):
        """Training routine with 5-fold cross-validation for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        full_dataset = data_loader["train"].dataset  # Assumes a standard PyTorch Dataset
        indices = list(range(len(full_dataset)))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold_idx, (train_indices, valid_indices) in enumerate(kf.split(indices)):
            print(f"\n=== Starting Fold {fold_idx + 1}/5 ===")

            train_subset = Subset(full_dataset, train_indices)
            valid_subset = Subset(full_dataset, valid_indices)

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)

            mean_training_losses = []
            mean_valid_losses = []
            lrs = []
            self.min_valid_loss = None
            self.best_epoch = 0

            self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=self.config.MODEL.PHYSNET.FRAME_NUM).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR,
                epochs=self.max_epoch_num, steps_per_epoch=len(train_loader))

            for epoch in range(self.max_epoch_num):
                print(f'\n==== Fold {fold_idx} Training Epoch: {epoch} ====')
                self.model.train()
                running_loss = 0.0
                train_loss = []
                tbar = tqdm(train_loader, ncols=80)

                for idx, batch in enumerate(tbar):
                    tbar.set_description(f"Train fold {fold_idx} epoch {epoch}")
                    data, label = batch[0].to(torch.float32).to(self.device), batch[1].to(torch.float32).to(self.device)
                    label = label[:, 1:2, :].squeeze(1)

                    rspo2, _, _, _ = self.model(data)
                    rmse_loss = 0.0
                    for bb in range(data.shape[0]):
                        rspo2_value = rspo2[bb] if isinstance(rspo2[bb], torch.Tensor) else torch.tensor(rspo2[bb].item(), device=label[bb].device)
                        label_value = label[bb].mean().float()
                        rmse_loss += torch.sqrt(F.mse_loss(rspo2_value, label_value))
                    rmse_loss /= data.shape[0]

                    loss = rmse_loss
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    running_loss += loss.item()
                    train_loss.append(loss.item())
                    lrs.append(self.scheduler.get_last_lr()[0])
                    tbar.set_postfix(loss=loss.item())

                mean_training_losses.append(np.mean(train_loss))

                valid_loss = self.valid_loader_eval(valid_loader)
                mean_valid_losses.append(valid_loss)
                print('Validation loss:', valid_loss)

                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.save_model(self.best_epoch, fold=fold_idx)
                    print(f"Update best model for Fold {fold_idx}! Best epoch: {self.best_epoch}")

            print(f"=== Fold {fold_idx} finished: Best Epoch = {self.best_epoch}, Min Val Loss = {self.min_valid_loss} ===")

    def valid_loader_eval(self, valid_loader):
        """Validation over a custom loader."""
        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            vbar = tqdm(valid_loader, ncols=80)
            for valid_batch in vbar:
                data, label = valid_batch[0].to(torch.float32).to(self.device), valid_batch[1].to(torch.float32).to(self.device)
                label = label[:, 1:2, :].squeeze(1)

                rspo2, _, _, _ = self.model(data)
                for bb in range(data.shape[0]):
                    rspo2_value = torch.tensor(rspo2[bb].item(), device=label[bb].device) if not isinstance(rspo2[bb], torch.Tensor) else rspo2[bb]
                    label_value = label[bb].mean().float()
                    valid_loss.append(F.mse_loss(rspo2_value, label_value))

        loss_tensor = torch.stack(valid_loss)
        return torch.sqrt(loss_tensor.mean())


    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            
            # Load the checkpoint
            checkpoint = torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device)
            
            # Check if the checkpoint is a dictionary and contains 'model_state_dict'
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)  # Load directly if it's just state_dict
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        test_loss = []
        rspo2_values = []
        label_values = []
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                label= np.squeeze(label[:,1:2,:],axis=1)
                rspo2, _, _, _ = self.model(data)
                # print(label.ndim)
                if label.ndim == 3:
                    label = np.squeeze(label[:, 1:2, :])
                # print(label)
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    rspo2 = rspo2.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = rspo2[idx]

                    rspo2_value = torch.tensor(rspo2[idx].item(), device=label[idx].device) if not isinstance(rspo2[idx], torch.Tensor) else rspo2[idx]
                    rounded_value = round(rspo2_value.item())
                    rspo2_value = torch.tensor(rounded_value, device=rspo2_value.device)
                    label_value = label[idx].mean().float().round()

                    rspo2_values.append(rspo2_value.item())
                    label_values.append(label_value.item())

                    test_loss.append(F.mse_loss(rspo2_value, label_value))
                    labels[subj_index][sort_index] = label[idx]

        print('')
        spo2_errors_tensor = torch.stack(test_loss)  # Stack into a single tensor
        RMSE = torch.sqrt(spo2_errors_tensor.mean())
        print(rspo2_values)
        # print(label_values)
        print("RMSE:", RMSE.item(), "\nPredicted SpO2 value:", np.mean(rspo2_values), "\nGround Truth value:", np.mean(label_values))
        # calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index, fold=0):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, f"{self.model_file_name}_Fold{fold}_Epoch{index}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, model_path)
        print('Saved Model Path:', model_path)
