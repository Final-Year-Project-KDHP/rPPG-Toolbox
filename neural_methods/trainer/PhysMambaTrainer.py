"""PhysMamba Trainer (SpO2 branch)."""
import os
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from evaluation.metrics import calculate_metrics
# from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson  # Not used for SpO2
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from scipy.signal import welch

class PhysMambaTrainer(BaseTrainer):

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
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        self.frame_rate = config.TRAIN.DATA.FS

        # Initialize model
        self.model = PhysMamba().to(self.device)
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        # ToolBox mode logic
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config.TRAIN.LR, 
                weight_decay=0.0005
            )
            # OneCycleLR scheduler
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches
            )
        elif config.TOOLBOX_MODE == "only_test":
            # No optimizer needed for only_test scenario,
            # unless you plan to load from checkpoint
            pass
        else:
            raise ValueError("PhysMambaTrainer initialized in incorrect toolbox mode!")

        # Optional: If continuing from a previous checkpoint
        if getattr(config.TRAIN, "CONTINUE_TRAIN", False):
            if not os.path.exists(config.INFERENCE.MODEL_PATH):
                raise ValueError("Checkpoint path does not exist for CONTINUE_TRAIN.")
            print("Loading checkpoint from:", config.INFERENCE.MODEL_PATH)
            checkpoint = torch.load(config.INFERENCE.MODEL_PATH, map_location=self.device)
            # If your checkpoint is saved with a dictionary structure:
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                # If the file is just the raw state_dict
                self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)

    def train(self, data_loader):
        """Training routine for model (SpO2)."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            running_loss = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")

                # Assumes batch[0]: input frames, batch[1]: spo2 label
                data, label = batch[0].float(), batch[1].float()
                data = data.to(self.device)
                label = label.to(self.device)

                # print("-----------------label shape:",label.shape)

                # Example: If label dimension is [N, 1, T] or [N, X, ...], 
                # and you only need a single scalar target, you might do:
                # label = label.mean(dim=(-1, -2, ...)) or
                # label = label.squeeze()  # Adjust as needed based on your data shape.
                # If your label is purely [N, 1], you might not need extra squeezing.

                label = label.mean(dim=1, keepdim=True)
                print("-----------------averaged label shape:",label.shape)

                label = label.squeeze()
                print("-----------------averaged squeezed label shape:",label.shape)


                self.optimizer.zero_grad()

                # Forward pass
                pred_spo2 = self.model(data)

                # If your model outputs shape [N] or [N, 1], unify them:
                if len(pred_spo2.shape) > 1:
                    pred_spo2 = pred_spo2.squeeze()
                print("-----------------pred spo2 squeezed:",pred_spo2.shape)


                # Calculate RMSE (you could do MSE; here, we demonstrate RMSE)
                mse_loss = torch.mean((pred_spo2 - label) ** 2)
                rmse_loss = torch.sqrt(mse_loss)

                rmse_loss.backward()
                running_loss += rmse_loss.item()

                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(loss=rmse_loss.item())

            self.save_model(epoch)

            # Validation logic
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None or (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch:", self.best_epoch)

            torch.cuda.empty_cache()

        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    def valid(self, data_loader):
        """ Runs validation. Typically, we'd compute an RMSE for SpO2. """
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validating===")
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data, label = valid_batch[0].float(), valid_batch[1].float()
                data = data.to(self.device)
                label = label.to(self.device)

                label = label.mean(dim=1, keepdim=True)
                label = label.squeeze()

                pred_spo2 = self.model(data)
                if len(pred_spo2.shape) > 1:
                    pred_spo2 = pred_spo2.squeeze()

                # RMSE
                mse_loss = torch.mean((pred_spo2 - label) ** 2)
                rmse_loss = torch.sqrt(mse_loss)
                valid_loss.append(rmse_loss.item())

                vbar.set_postfix(loss=rmse_loss.item())

        # Return average RMSE
        return float(np.mean(valid_loss))

    def test(self, data_loader):
        """ Runs the model on test sets for SpO2. """
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        # Load model if we're in test mode
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)

            checkpoint = torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device))
            else:
                best_model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):

                print("-------------debugging test_batch:",test_batch)

                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.device), test_batch[1].to(self.device)

                label = label.mean(dim=1, keepdim=True)# mean over the channel dimension
                print("-----------------averaged label :",label)


                # Forward pass
                pred_spo2_test = self.model(data)
                if len(pred_spo2_test.shape) > 1:
                    pred_spo2_test = pred_spo2_test.squeeze()

                # Optionally move data to CPU for saving
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu().squeeze()
                    pred_spo2_test = pred_spo2_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()

                    print("-------------debugging pred_spo2_test:",pred_spo2_test)

                    # You can store these in predictions/labels for further analysis
                    predictions[subj_index][sort_index] = pred_spo2_test[idx]
                    labels[subj_index][sort_index] = label[idx].squeeze()

        print('')
        # If you have a custom SpO2-based metric, you could call it here
        calculate_metrics(predictions, labels, self.config)

        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """Saves both model and optimizer state_dict if desired."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        
        # If you'd like to save just the model weights (like in the main branch):
        # torch.save(self.model.state_dict(), model_path)
        # Otherwise, if you want the optimizer as well:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, model_path)
        print('Saved Model Path: ', model_path)

    # Keeping the HR calculation function as is, though it may not be relevant for SpO2
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p > min/60) & (p < max/60)][
            np.argmax(q[(p > min/60) & (p < max/60)])
        ] * 60
