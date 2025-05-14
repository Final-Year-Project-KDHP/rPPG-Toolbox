"""PhysNet Trainer."""
print
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import plot_label_distribution, compute_lds_weights_with_plots
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn




class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        print("device:", self.device)
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
            # self.loss_fn = self.custom_loss  # You can adjust delta (default is 1.0)

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
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        # plot_label_distribution(
        #     data_loader=data_loader["train"],
        #     save_dir=self.config.LOG.PATH,
        #     filename="label_distribution.png"
        # )

        if self.config.TRAIN.CONTINUE_TRAIN:
            print("Loading checkpoint")
            checkpoint = torch.load(self.config.INFERENCE.MODEL_PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model = self.model.to(self.config.DEVICE)
            # for layer in [self.model.ConvBlock1, self.model.ConvBlock2]:
            #     for param in layer.parameters():
            #         param.requires_grad = False


        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            count=0
            for idx, batch in enumerate(tbar):
              tbar.set_description(f"Train epoch {epoch}")
              data, label, filename = batch[0].to(torch.float32).to(self.device), \
                                      batch[1].to(torch.float32).to(self.device), \
                                      batch[2]
              label = batch[1][:, 1:2, :].squeeze(1).to(dtype=torch.float32, device=self.device)
              mean_label = label.mean(dim=1)
              mask = (mean_label >= 90) & (mean_label <= 100)
              if mask.sum() == 0:
                continue  # Skip this batch if no valid samples
              data, label, mean_label = data[mask], label[mask], mean_label[mask]


              # Forward + Backward: concatenate along batch dimension
              reversed_data = torch.flip(data, dims=[2])  # flip along time dimension
              reversed_label = torch.flip(label, dims=[1])  # same flip for label
              reversed_mean_label = reversed_label.mean(dim=1)

              # Merge forward + backward
              data_combined = torch.cat([data, reversed_data], dim=0)
              label_combined = torch.cat([label, reversed_label], dim=0)
              mean_label_combined = torch.cat([mean_label, reversed_mean_label], dim=0)
            
              mean_label_np = mean_label_combined.detach().cpu().numpy().tolist()
              lds_weights = compute_lds_weights_with_plots(save_dir=self.config.LOG.PATH,mean_labels=mean_label_np)
              lds_weights = torch.tensor(lds_weights, dtype=torch.float32, device=self.device)

              rspo2, x_visual, x_visual3232, x_visual1616 = self.model(data_combined)
            
              sample_loss = F.mse_loss(rspo2.squeeze(), mean_label_combined, reduction='none')
              weighted_loss = sample_loss * lds_weights
              loss = torch.sqrt(weighted_loss.mean())

            #   loss = F.mse_loss(rspo2.squeeze(), mean_label_combined, reduction='mean')  # pure MSE
            #   loss = torch.sqrt(loss)
              loss.backward()
              running_loss += loss.item()
              train_loss.append(loss.item())

              # Optimizer step
              self.optimizer.step()
              self.scheduler.step()
              lrs.append(self.scheduler.get_last_lr()[0]) 
              self.optimizer.zero_grad()  # Reset gradients
              tbar.set_postfix(loss=loss.item())
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("\n ====Validing===")
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data, label = valid_batch[0].to(torch.float32).to(self.device), \
                              valid_batch[1].to(torch.float32).to(self.device)
                label = label[:, 1:2, :].squeeze(1)
                mean_label = label.mean(dim=1)

                mask = (mean_label >= 90) & (mean_label <= 100)
                if mask.sum() == 0:
                    continue
                data, mean_label = data[mask], mean_label[mask]

                rspo2, *_ = self.model(data)
                # loss = F.mse_loss(rspo2.squeeze(), mean_label, reduction='none')
                loss = F.l1_loss(rspo2.squeeze(), mean_label, reduction='none')  # MAE
                valid_loss.append(loss)

            if len(valid_loss) == 0:
                return torch.tensor(float('nan'))

            all_losses = torch.cat(valid_loss)
            RMSE = torch.sqrt(all_losses.mean())

        return RMSE

        


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
        test_mae=[]
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
                    label_mean = label[idx].mean().item()
                    
                    # print(label_mean)
                    if label_mean < 90 or label_mean > 100:
                        continue  # Skip samples where mean SpO2 < 90
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
                    test_mae.append(F.l1_loss(rspo2_value, label_value))
                    labels[subj_index][sort_index] = label[idx]

        print('')
        spo2_errors_tensor = torch.stack(test_loss)  # Stack into a single tensor
        RMSE = torch.sqrt(spo2_errors_tensor.mean())
        print(rspo2_values)
        # print(label_values)
        print("RMSE:", RMSE.item(), "\nPredicted SpO2 value:", np.mean(rspo2_values), "\nGround Truth value:", np.mean(label_values))
        MAE= torch.mean(torch.stack(test_mae))
        print("MAE:", MAE.item())
        # calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save({"model_state_dict":self.model.state_dict(),
                    "optimizer_state_dict":self.optimizer.state_dict()}, model_path)
        print('Saved Model Path: ', model_path)
    
    