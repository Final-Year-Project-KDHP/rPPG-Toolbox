"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.CustomCrossEntropyLoss import CustomCrossEntropyWithSelectivePenalty
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm


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
            #self.loss_model = Neg_Pearson()
            self.ce_penalty_loss_fn = CustomCrossEntropyWithSelectivePenalty(alpha=0.5)
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
        """Training routine for the model."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        if self.config.TRAIN.CONTINUE_TRAIN:
            print("Loading checkpoint")
            checkpoint = torch.load(self.config.INFERENCE.MODEL_PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model = self.model.to(self.config.DEVICE)

        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # Load data and labels
                data, labels = batch[0].to(torch.float32).to(self.device), batch[1].to(self.device)

                # Forward pass
                logits, rspo2, _, _, _ = self.model(data)

                # Initialize loss for the batch
                batch_loss = 0.0

                # Compute loss for each sample in the batch
                for bb in range(data.shape[0]):
                    rspo2_value = torch.tensor(rspo2[bb].item(), device=labels[bb].device) if not isinstance(rspo2[bb], torch.Tensor) else rspo2[bb]
                    label_value = labels[bb].mean().float()
                    #map label to class
                    label_value = self.map_to_class(label_value.item())
                    sample_loss = self.ce_penalty_loss_fn(rspo2_value, label_value)
                    batch_loss += sample_loss

                # Average loss across the batch
                batch_loss /= data.shape[0]

                # Backward pass and optimization
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update running loss
                running_loss += batch_loss.item()
                train_loss.append(batch_loss.item())

                # Logging
                if idx % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

                tbar.set_postfix(loss=batch_loss.item())

            mean_training_losses.append(np.mean(train_loss))

            # Save model and validate
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print("Validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

        # Plot losses and learning rates
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)



    def valid(self, data_loader):
        """Runs the model on validation sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for validation")

        print("====Validating===")
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                # Load data and labels
                data, labels = valid_batch[0].to(torch.float32).to(self.device), valid_batch[1].to(self.device)

                # Forward pass
                logits, rspo2, _, _, _ = self.model(data)

                # Compute loss for each sample in the batch
                batch_loss = 0.0
                for bb in range(data.shape[0]):
                    rspo2_value = torch.tensor(rspo2[bb].item(), device=labels[bb].device) if not isinstance(rspo2[bb], torch.Tensor) else rspo2[bb]
                    label_value = labels[bb].mean().float()
                    label_value = self.map_to_class(label_value.item())
                    sample_loss = self.ce_penalty_loss_fn(rspo2_value, label_value)
                    batch_loss += sample_loss.item()

                # Append the mean loss for the batch
                valid_loss.append(batch_loss / data.shape[0])

            # Compute mean validation loss
            mean_valid_loss = np.mean(valid_loss)
            return mean_valid_loss



    def test(self, data_loader):
        """Runs the model on test sets using CrossEntropyLoss."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        # Load the appropriate model checkpoint
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH)["model_state_dict"])
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path)["model_state_dict"])
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path)["model_state_dict"])

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        test_losses = []

        cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                rspo2, _, _, _ = self.model(data)

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
                    label_value = label[idx].mean().float()
                    label_value = self.map_to_class(label_value.item())
                    test_losses.append(cross_entropy_loss_fn(rspo2_value, label_value))
                    labels[subj_index][sort_index] = label[idx]

        # Compute average test loss
        mean_test_loss = np.mean(test_losses)
        print(f"Mean Test Loss (CrossEntropy): {mean_test_loss:.4f}")

        # Save test outputs if configured
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

        return mean_test_loss


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save({"model_state_dict":self.model.state_dict(),
                    "optimizer_state_dict":self.optimizer.state_dict()}, model_path)
        print('Saved Model Path: ', model_path)
    
    def map_to_class(self, value):
        """Maps SpO2 values to class indices."""
        if value < 90:
            return 0
        elif value > 100:
            return 12
        else:
            return int(value)-89
