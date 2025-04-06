"""PhysMamba Multi-Task Trainer."""
import os
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import welch

# from neural_methods.model.PhysMambaMultiTask import PhysMambaMultiTask  # <-- import your multi-task model
from neural_methods.model.PhysMamba import PhysMambaMultiTask  # <-- import your multi-task model
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson  # your existing negative Pearson
from evaluation.metrics import calculate_metrics  # or your custom metric function(s)


class PhysMambaMultiTaskTrainer(BaseTrainer):
    """
    Trains and tests a multi-task PhysMamba model that outputs both rPPG/HR and SpO2.
    """

    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Basic training hyperparameters
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.frame_rate = config.TRAIN.DATA.FS  # if needed
        self.min_valid_loss = None
        self.best_epoch = 0

        # Initialize loss and LR history for plotting
        self.train_loss_history = []
        self.valid_loss_history = []
        self.lr_history = []

        # Model
        self.model = PhysMambaMultiTask(
            theta=0.5,
            drop_rate1=0.25,
            drop_rate2=0.5,
            frames=128  # or set from config if needed
        ).to(self.device)

        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(self.num_of_gpu))
            )

        # Mode logic
        if config.TOOLBOX_MODE == "train_and_test":
            # Optimizer & LR Scheduler
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0005
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=self.max_epoch_num,
                steps_per_epoch=self.num_train_batches
            )

            # Losses
            self.criterion_hr = Neg_Pearson()   # for HR
            # For SpO2, we use RMSE. You can define a custom criterion if you prefer:
            self.criterion_spo2 = lambda preds, targets: torch.sqrt(
                torch.mean((preds - targets) ** 2)
            )

        elif config.TOOLBOX_MODE == "only_test":
            # In test mode, we just create the model; no need for optimizer/scheduler
            pass
        else:
            raise ValueError("Incorrect toolbox mode for multi-task trainer!")

    def train(self, data_loader):
        """Training routine for the multi-task model."""
        if data_loader["train"] is None:
            raise ValueError("No data for training.")

        for epoch in range(self.max_epoch_num):
            print("\n====Training Epoch: {}====".format(epoch))
            self.model.train()
            running_loss = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch {}".format(epoch))

                data = batch[0].float().to(self.device)   # [B, 3, T, H, W]
                label = batch[1].float().to(self.device)     # [B, 2, T] (assuming channel 0=HR, 1=SpO2)
                
                # 1) Extract the HR signal
                hr_label = label[:, 0, :]  # shape [B, T]
                
                # 2) Extract the SpO2 label
                #    If your SpO2 label is also a time series, you might do something else:
                #    spo2_label = label[:, 1, :] and then reduce to a single scalar if needed.
                #    Example: single scalar is average across T:
                spo2_label = label[:, 1, :].mean(dim=-1)  # shape [B]

                # Forward pass
                rppg_pred, spo2_pred = self.model(data)   # rppg_pred: [B, T], spo2_pred: [B, 1]

                # Some normalizations for HR (optional):
                rppg_pred = (rppg_pred - rppg_pred.mean(dim=-1, keepdim=True)) / (rppg_pred.std(dim=-1, keepdim=True) + 1e-6)
                hr_label = (hr_label - hr_label.mean(dim=-1, keepdim=True)) / (hr_label.std(dim=-1, keepdim=True) + 1e-6)

                # SpO2_pred might be shape [B, 1]. Let's squeeze to [B].
                spo2_pred = spo2_pred.squeeze(-1)

                # Compute losses
                hr_loss = self.criterion_hr(rppg_pred, hr_label)        # Negative Pearson
                spo2_loss = self.criterion_spo2(spo2_pred, spo2_label)    # RMSE

                total_loss = hr_loss + spo2_loss  # Weighted sum if needed

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += total_loss.item()
                tbar.set_postfix(
                    total=total_loss.item(),
                    hr=hr_loss.item(),
                    spo2=spo2_loss.item()
                )

            avg_loss = running_loss / len(data_loader["train"])
            print(f"Epoch [{epoch}] Avg Train Loss: {avg_loss:.4f}")
            self.train_loss_history.append(avg_loss)

            # Record current learning rate (OneCycleLR updates it every step)
            current_lr = self.scheduler.get_last_lr()[0]
            self.lr_history.append(current_lr)

            # Save model at the end of each epoch
            self.save_model(epoch)

            # Validation
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print("Validation Loss: ", valid_loss)
                self.valid_loss_history.append(valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch:", self.best_epoch)

            # Call the plotting function if enabled in the configuration
            if self.config.TRAIN.PLOT_LOSSES_AND_LR:
                self.plot_losses_and_lrs(
                    train_loss=self.train_loss_history,
                    valid_loss=self.valid_loss_history,
                    lrs=self.lr_history,
                    config=self.config
                )

            torch.cuda.empty_cache()

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss
            ))

    def valid(self, data_loader):
        """Validation routine for the multi-task model."""
        if data_loader["valid"] is None:
            raise ValueError("No data for validation.")
        print("\n====Validating====")
        self.model.eval()

        total_losses = []
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(vbar):
                data = batch[0].float().to(self.device)
                label = batch[1].float().to(self.device)  # [B, 2, T]

                # HR label
                hr_label = label[:, 0, :]
                # SpO2 label
                spo2_label = label[:, 1, :].mean(dim=-1)

                rppg_pred, spo2_pred = self.model(data)
                spo2_pred = spo2_pred.squeeze(-1)

                # Optional normalization for HR
                rppg_pred = (rppg_pred - rppg_pred.mean(dim=-1, keepdim=True)) / (rppg_pred.std(dim=-1, keepdim=True) + 1e-6)
                hr_label = (hr_label - hr_label.mean(dim=-1, keepdim=True)) / (hr_label.std(dim=-1, keepdim=True) + 1e-6)

                hr_loss = Neg_Pearson()(rppg_pred, hr_label)
                spo2_loss = torch.sqrt(torch.mean((spo2_pred - spo2_label) ** 2))

                total_loss = hr_loss + spo2_loss
                total_losses.append(total_loss.item())

                vbar.set_postfix(
                    total=total_loss.item(),
                    hr=hr_loss.item(),
                    spo2=spo2_loss.item()
                )

        return float(np.mean(total_losses))

    def test(self, data_loader):
        """Test routine for the multi-task model. Predicts HR waveform and SpO2."""
        if data_loader["test"] is None:
            raise ValueError("No data for test.")
        print("\n===Testing===")

        # Depending on your config, load the best epoch or last epoch
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Check INFERENCE.MODEL_PATH.")
            print("Loading pretrained model:", self.config.INFERENCE.MODEL_PATH)
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_path = os.path.join(
                    self.model_dir, self.model_file_name + "_Epoch" + str(self.max_epoch_num - 1) + ".pth"
                )
                print("Testing uses last epoch model:", last_epoch_path)
                self.model.load_state_dict(torch.load(last_epoch_path))
            else:
                best_epoch_path = os.path.join(
                    self.model_dir, self.model_file_name + "_Epoch" + str(self.best_epoch) + ".pth"
                )
                print("Testing uses best epoch model:", best_epoch_path)
                self.model.load_state_dict(torch.load(best_epoch_path))

        self.model.eval()
        self.model.to(self.device)

        # We'll store predictions in dictionaries for later metric calculations
        hr_predictions = {}
        hr_labels = {}
        spo2_predictions = {}
        spo2_labels = {}

        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for _, test_batch in enumerate(tbar):
                data = test_batch[0].to(self.device)
                label = test_batch[1].to(self.device)  # [B, 2, T]

                # In your dataset, test_batch[2] might contain subject IDs, test_batch[3] might contain sort indices
                subject_ids = test_batch[2]
                sort_indices = test_batch[3]

                # Forward pass
                rppg_pred, spo2_pred = self.model(data)
                spo2_pred = spo2_pred.squeeze(-1)  # [B]

                # Optionally, store them
                for idx in range(data.shape[0]):
                    subj_id = subject_ids[idx]
                    sort_idx = int(sort_indices[idx])

                    if subj_id not in hr_predictions:
                        hr_predictions[subj_id] = {}
                        hr_labels[subj_id] = {}
                        spo2_predictions[subj_id] = {}
                        spo2_labels[subj_id] = {}

                    # HR
                    hr_predictions[subj_id][sort_idx] = rppg_pred[idx].cpu()
                    hr_labels[subj_id][sort_idx] = label[idx, 0, :].cpu()

                    # SpO2
                    spo2_predictions[subj_id][sort_idx] = spo2_pred[idx].cpu()
                    # Typically the ground-truth SpO2 is label[:,1,:].mean(dim=-1)
                    gt_spo2 = label[idx, 1, :].mean().cpu()
                    spo2_labels[subj_id][sort_idx] = gt_spo2

        # Now you can run your own custom metrics on hr_predictions/spo2_predictions vs. hr_labels/spo2_labels
        # For example:
        # calculate_metrics(hr_predictions, hr_labels, self.config)
        # calculate_metrics(spo2_predictions, spo2_labels, self.config)

        # Optionally save outputs
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs((hr_predictions, spo2_predictions),
                                   (hr_labels, spo2_labels),
                                   self.config)

    def save_model(self, epoch):
        """Save the model (and possibly optimizer state) to disk."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, f"{self.model_file_name}_Epoch{epoch}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path:", model_path)

    def get_hr(self, y, sr=30, min=30, max=180):
        """
        Compute HR from a waveform y via Welch's method. 
        This function is taken from your original trainer for reference.
        """
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p > min/60) & (p < max/60)][np.argmax(q[(p > min/60) & (p < max/60)])] * 60
