"""PhysNet Trainer with 5-Fold Cross-Validation."""

import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer


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
        self.config = config

        self.data_loader = data_loader
        self.min_valid_loss = None
        self.best_epoch = 0
        self.best_model_state = None  # Store best model across folds

    def build_model(self):
        model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=self.config.MODEL.PHYSNET.FRAME_NUM
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.TRAIN.LR, 
            epochs=self.config.TRAIN.EPOCHS, 
            steps_per_epoch=self.num_train_batches_per_fold
        )
        return model, optimizer, scheduler

    def train(self):
        """Training routine with 5-Fold Cross-Validation."""
        if self.data_loader["train"] is None:
            raise ValueError("No data for train")

        dataset = self.data_loader["train"].dataset
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_losses = []

        print(f"Starting 5-Fold Cross-Validation")
        indices = np.arange(len(dataset))

        for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(indices)):
            print(f"\n=== Fold {fold_idx+1}/5 ===")

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, sampler=train_sampler)
            valid_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, sampler=valid_sampler)

            self.num_train_batches_per_fold = len(train_loader)

            # Initialize new model and optimizer for each fold
            self.model, self.optimizer, self.scheduler = self.build_model()
            self.loss_model = Neg_Pearson()

            mean_training_losses = []
            mean_valid_losses = []
            lrs = []

            self.min_valid_loss = None

            for epoch in range(self.max_epoch_num):
                print(f'\n--- Training Epoch {epoch} (Fold {fold_idx+1}) ---')
                running_loss = 0.0
                train_loss = []

                self.model.train()
                tbar = tqdm(train_loader, ncols=80)

                for idx, batch in enumerate(tbar):
                    tbar.set_description(f"Train Epoch {epoch} Fold {fold_idx+1}")
                    data, label, _ = batch[0].to(torch.float32).to(self.device), \
                                     batch[1].to(torch.float32).to(self.device), \
                                     batch[2]
                    label = batch[1][:, 1:2, :].squeeze(1).to(dtype=torch.float32, device=self.device)

                    rspo2, _, _, _ = self.model(data)

                    rmse_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    for bb in range(data.shape[0]):
                        rspo2_value = rspo2[bb] if isinstance(rspo2[bb], torch.Tensor) else torch.tensor(rspo2[bb].item(), device=label[bb].device, dtype=torch.float32)
                        label_value = label[bb].mean().float()
                        rmse_loss += torch.sqrt(F.mse_loss(rspo2_value, label_value))
                    rmse_loss /= data.shape[0]

                    loss = rmse_loss
                    loss.backward()

                    running_loss += loss.item()
                    train_loss.append(loss.item())

                    self.optimizer.step()
                    self.scheduler.step()
                    lrs.append(self.scheduler.get_last_lr()[0])
                    self.optimizer.zero_grad()
                    tbar.set_postfix(loss=loss.item())

                mean_training_losses.append(np.mean(train_loss))

                valid_loss = self.valid(valid_loader)
                mean_valid_losses.append(valid_loss)
                print(f'Validation Loss after Epoch {epoch}: {valid_loss:.6f}')

                # Save the best model across all folds
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    self.best_model_state = self.model.state_dict()
                    print(f"New best model found at Fold {fold_idx+1}, Epoch {epoch}")

            fold_losses.append(self.min_valid_loss)

        # After all folds
        print(f"\n=== Cross Validation Complete ===")
        print(f"Average Validation Loss across 5 folds: {np.mean(fold_losses):.6f}")

        # Save the best model across all folds
        self.save_model()

    def valid(self, valid_loader):
        """Validation function."""
        valid_loss = []
        self.model.eval()

        with torch.no_grad():
            vbar = tqdm(valid_loader, ncols=80)
            for valid_batch in vbar:
                vbar.set_description("Validation")
                data, label, _ = valid_batch[0].to(torch.float32).to(self.device), \
                                 valid_batch[1].to(torch.float32).to(self.device), \
                                 valid_batch[2]
                label = label[:, 1:2, :].squeeze(1)

                rspo2, _, _, _ = self.model(data)

                for bb in range(data.shape[0]):
                    rspo2_value = rspo2[bb] if isinstance(rspo2[bb], torch.Tensor) else torch.tensor(rspo2[bb].item(), device=label.device, dtype=torch.float32)
                    label_value = label[bb].mean().float()
                    valid_loss.append(F.mse_loss(rspo2_value, label_value))

            spo2_errors_tensor = torch.stack(valid_loss)
            RMSE = torch.sqrt(spo2_errors_tensor.mean())
        return RMSE.item()

    def save_model(self):
        """Save the best model from cross-validation."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        save_path = os.path.join(
            self.model_dir, self.model_file_name + '_BestCrossVal.pth'
        )
        torch.save(self.best_model_state, save_path)
        print(f"Best model saved at: {save_path}")
