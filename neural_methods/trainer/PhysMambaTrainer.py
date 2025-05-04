"""PhysMamba Multi-Task Trainer."""
import os
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# from neural_methods.model.PhysMambaMultiTask import PhysMambaMultiTask  # <-- import your multi-task model
from neural_methods.model.PhysMamba import PhysMambaMultiTask  # <-- import your multi-task model
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson  # your existing negative Pearson
from neural_methods.loss.FrequencyHybrid import frequency_loss_waveform

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

        self.d   = getattr(config.TRAIN, "W_NEG_PEARSON", 0.2)
        self.w_freq = getattr(config.TRAIN, "W_FREQ",        0.8)
        self.diff_flag = (config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized")
        self.fs        = config.TRAIN.DATA.FS

        # Initialize histories for overall loss and learning rate
        self.train_loss_history = []
        self.valid_loss_history = []
        self.lr_history = []

        # Initialize histories for task-specific losses: HR and SpO2 (for training and validation)
        self.hr_loss_history = []
        self.spo2_loss_history = []
        self.valid_hr_loss_history = []
        self.valid_spo2_loss_history = []
        # track lambda if desired
        self.lambda_history = []

        # Model
        self.model = PhysMambaMultiTask(
            theta=0.5,
            drop_rate1=0.25,
            drop_rate2=0.5,
            frames=128, # or set from config if needed
            learnable_balance  = getattr(config.TRAIN, "LEARNABLE_BALANCE", False),
            init_lambda        = getattr(config.TRAIN, "INIT_LAMBDA", 0.5),
            cross_fuse_cfg = config.MODEL.CROSS_FUSE,
            k_round        = config.MODEL.ROUNDING_SIGMOID.K
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
            # For SpO2, we use RMSE.
            self.criterion_spo2 = lambda preds, targets: torch.sqrt(
                torch.mean((preds - targets) ** 2)
            )

        elif config.TOOLBOX_MODE == "only_test":
            # In test mode, we just create the model; no need for optimizer/scheduler
            pass
        else:
            raise ValueError("Incorrect toolbox mode for multi-task trainer!")

    def plot_task_losses(self, hr_train, hr_valid, spo2_train, spo2_valid, config):
        """Plot and save HR loss and SpO2 loss in separate PDF files with validation legends (in yellow)."""
        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Plotting of losses only supports train_and_test mode!')

        epochs = range(0, len(hr_train))

        # Plot HR Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, hr_train, label='Training HR Loss')
        plt.plot(epochs, hr_valid, color='yellow', label='Validation HR Loss')
        plt.xlabel('Epoch')
        plt.ylabel('HR Loss')
        plt.title(f'{filename_id} HR Losses')
        plt.legend()
        plt.xticks(epochs)
        hr_loss_plot_filename = os.path.join(output_dir, f"{filename_id}_hr_losses.pdf")
        plt.savefig(hr_loss_plot_filename, dpi=300)
        plt.close()

        # Plot SpO2 Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, spo2_train, label='Training SpO2 Loss')
        plt.plot(epochs, spo2_valid, color='yellow', label='Validation SpO2 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('SpO2 Loss')
        plt.title(f'{filename_id} SpO2 Losses')
        plt.legend()
        plt.xticks(epochs)
        spo2_loss_plot_filename = os.path.join(output_dir, f"{filename_id}_spo2_losses.pdf")
        plt.savefig(spo2_loss_plot_filename, dpi=300)
        plt.close()

        print('Saved HR Loss plot to:', hr_loss_plot_filename)
        print('Saved SpO2 Loss plot to:', spo2_loss_plot_filename)

    def train(self, data_loader):
        """Training routine for the multi-task model."""
        if data_loader["train"] is None:
            raise ValueError("No data for training.")

        for epoch in range(self.max_epoch_num):
            print("\n====Training Epoch: {}====".format(epoch))
            self.model.train()
            running_loss = 0.0
            running_hr_loss = 0.0
            running_spo2_loss = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch {}".format(epoch))

                data = batch[0].float().to(self.device)   # [B, 3, T, H, W]
                label = batch[1].float().to(self.device)    # [B, 2, T] (channel 0=HR, 1=SpO2)

                # Extract HR signal and SpO2 label (average over T)
                hr_label = label[:, 0, :]  # shape [B, T]
                spo2_label = label[:, 1, :]  # [B, T]

                # Forward pass
                rppg_pred, spo2_pred, λ = self.model(data)   # rppg_pred: [B, T], spo2_pred: [B, T]

                # Optional normalization for HR
                rppg_pred = (rppg_pred - rppg_pred.mean(dim=-1, keepdim=True)) / (rppg_pred.std(dim=-1, keepdim=True) + 1e-6)
                hr_label = (hr_label - hr_label.mean(dim=-1, keepdim=True)) / (hr_label.std(dim=-1, keepdim=True) + 1e-6)

                # Squeeze SpO2 predictions if needed
                # spo2_pred = spo2_pred.squeeze(-1)

                # Compute losses
                # hr_loss = self.criterion_hr(rppg_pred, hr_label)        # Negative Pearson for HR

                loss_np = self.criterion_hr(rppg_pred, hr_label)          # scalar
                loss_freq, aux = frequency_loss_waveform(
                        pred_wave   = rppg_pred,
                        gt_wave     = hr_label,
                        Fs          = self.fs,
                        diff_flag   = self.diff_flag,
                        std         = 3.0,         # or expose in YAML
                        tau         = 1.5
                        )

                hr_loss  = self.w_np * loss_np + self.w_freq * loss_freq

                spo2_loss = self.criterion_spo2(spo2_pred, spo2_label)    # RMSE for SpO2
                total_loss = λ * hr_loss + (1.0 - λ) * spo2_loss

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += total_loss.item()
                running_hr_loss += hr_loss.item()
                running_spo2_loss += spo2_loss.item()

                tbar.set_postfix(
                    total=total_loss.item(),
                    hr=hr_loss.item(),
                    spo2=spo2_loss.item(),
                    λ     = λ.item()
                )

            avg_loss = running_loss / len(data_loader["train"])
            avg_hr_loss = running_hr_loss / len(data_loader["train"])
            avg_spo2_loss = running_spo2_loss / len(data_loader["train"])

            print(f"Epoch [{epoch}] Avg Train Loss: {avg_loss:.4f}")
            self.train_loss_history.append(avg_loss)
            self.hr_loss_history.append(avg_hr_loss)
            self.spo2_loss_history.append(avg_spo2_loss)

            # Record current learning rate (OneCycleLR updates it every step)
            current_lr = self.scheduler.get_last_lr()[0]
            self.lr_history.append(current_lr)

            # Save model at the end of each epoch
            self.save_model(epoch)

            # Validation
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, valid_hr_loss, valid_spo2_loss = self.valid(data_loader)
                print("Validation Loss: ", valid_loss)
                self.valid_loss_history.append(valid_loss)
                self.valid_hr_loss_history.append(valid_hr_loss)
                self.valid_spo2_loss_history.append(valid_spo2_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch:", self.best_epoch)

            # Call plotting functions if enabled in the configuration
            if self.config.TRAIN.PLOT_LOSSES_AND_LR:
                # Plot overall losses and learning rate (via BaseTrainer's method)
                self.plot_losses_and_lrs(
                    train_loss=self.train_loss_history,
                    valid_loss=self.valid_loss_history,
                    lrs=self.lr_history,
                    config=self.config
                )
                # Plot task-specific HR and SpO2 losses with validation curves in yellow
                self.plot_task_losses(
                    hr_train=self.hr_loss_history,
                    hr_valid=self.valid_hr_loss_history,
                    spo2_train=self.spo2_loss_history,
                    spo2_valid=self.valid_spo2_loss_history,
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
        valid_hr_losses = []
        valid_spo2_losses = []

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(vbar):
                data = batch[0].float().to(self.device)
                label = batch[1].float().to(self.device)  # [B, 2, T]

                # Extract HR signal and SpO2 label (average over T)
                hr_label = label[:, 0, :]
                spo2_label = label[:, 1, :]

                rppg_pred, spo2_pred,λ = self.model(data)
                # spo2_pred = spo2_pred.squeeze(-1)

                # Optional normalization for HR
                rppg_pred = (rppg_pred - rppg_pred.mean(dim=-1, keepdim=True)) / (rppg_pred.std(dim=-1, keepdim=True) + 1e-6)
                hr_label = (hr_label - hr_label.mean(dim=-1, keepdim=True)) / (hr_label.std(dim=-1, keepdim=True) + 1e-6)

                # hr_loss = Neg_Pearson()(rppg_pred, hr_label)
                # ─ HR branch ─────────────────────────────────────────
                # rppg_pred, hr_label normalised as before
                loss_np = self.criterion_hr(rppg_pred, hr_label)          # scalar
                loss_freq, aux = frequency_loss_waveform(
                        pred_wave   = rppg_pred,
                        gt_wave     = hr_label,
                        Fs          = self.fs,
                        diff_flag   = self.diff_flag,
                        std         = 3.0,         # or expose in YAML
                        tau         = 1.5
                        )

                hr_loss  = self.w_np * loss_np + self.w_freq * loss_freq

                spo2_loss = torch.sqrt(torch.mean((spo2_pred - spo2_label) ** 2))
                total_loss = λ * hr_loss + (1.0 - λ) * spo2_loss

                total_losses.append(total_loss.item())
                valid_hr_losses.append(hr_loss.item())
                valid_spo2_losses.append(spo2_loss.item())

                vbar.set_postfix(
                    total=total_loss.item(),
                    hr=hr_loss.item(),
                    spo2=spo2_loss.item(), 
                    λ=λ.item()
                )

        avg_total_loss = float(np.mean(total_losses))
        avg_hr_loss = float(np.mean(valid_hr_losses))
        avg_spo2_loss = float(np.mean(valid_spo2_losses))
        return avg_total_loss, avg_hr_loss, avg_spo2_loss

    # def test(self, data_loader):
    #     """Test routine for the multi-task model. Predicts HR waveform and SpO2."""
    #     if data_loader["test"] is None:
    #         raise ValueError("No data for test.")
    #     print("\n===Testing===")

    #     # Depending on your config, load best epoch or last epoch
    #     if self.config.TOOLBOX_MODE == "only_test":
    #         if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
    #             raise ValueError("Inference model path error! Check INFERENCE.MODEL_PATH.")
    #         print("Loading pretrained model:", self.config.INFERENCE.MODEL_PATH)
    #         self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
    #     else:
    #         if self.config.TEST.USE_LAST_EPOCH:
    #             last_epoch_path = os.path.join(
    #                 self.model_dir, self.model_file_name + "_Epoch" + str(self.max_epoch_num - 1) + ".pth"
    #             )
    #             print("Testing uses last epoch model:", last_epoch_path)
    #             self.model.load_state_dict(torch.load(last_epoch_path))
    #         else:
    #             best_epoch_path = os.path.join(
    #                 self.model_dir, self.model_file_name + "_Epoch" + str(self.best_epoch) + ".pth"
    #             )
    #             print("Testing uses best epoch model:", best_epoch_path)
    #             self.model.load_state_dict(torch.load(best_epoch_path))

    #     self.model.eval()
    #     self.model.to(self.device)

    #     # We'll store predictions in dictionaries for later metric calculations
    #     hr_predictions = {}
    #     hr_labels = {}
    #     spo2_predictions = {}
    #     spo2_labels = {}

    #     with torch.no_grad():
    #         tbar = tqdm(data_loader["test"], ncols=80)
    #         for _, test_batch in enumerate(tbar):
    #             data = test_batch[0].to(self.device)
    #             label = test_batch[1].to(self.device)  # [B, 2, T]

    #             # Assume test_batch[2] contains subject IDs and test_batch[3] contains sort indices
    #             subject_ids = test_batch[2]
    #             sort_indices = test_batch[3]

    #             # Forward pass
    #             rppg_pred, spo2_pred,_ = self.model(data)
    #             # spo2_pred = spo2_pred.squeeze(-1)  # [B]

    #             for idx in range(data.shape[0]):
    #                 subj_id = subject_ids[idx]
    #                 sort_idx = int(sort_indices[idx])

    #                 if subj_id not in hr_predictions:
    #                     hr_predictions[subj_id] = {}
    #                     hr_labels[subj_id] = {}
    #                     spo2_predictions[subj_id] = {}
    #                     spo2_labels[subj_id] = {}

    #                 hr_predictions[subj_id][sort_idx] = rppg_pred[idx].cpu()
    #                 hr_labels[subj_id][sort_idx] = label[idx, 0, :].cpu()
    #                 spo2_predictions[subj_id][sort_idx] = spo2_pred[idx].cpu()
    #                 spo2_labels[subj_id][sort_idx]      = label[idx, 1, :].cpu() # [T]

    #     # Optionally save outputs or calculate metrics
    #     if self.config.TEST.OUTPUT_SAVE_DIR:
    #         self.save_test_outputs((hr_predictions, spo2_predictions),
    #                                (hr_labels, spo2_labels),
    #                                self.config)

    # ------------------------------------------------------------

#   ‑‑ HR‑only evaluation  (SpO₂ ignored for now)
# ------------------------------------------------------------
    def test(self, data_loader):
        """Evaluate the trained model on the test set (HR branch only)."""
        if data_loader["test"] is None:
            raise ValueError("No data for test.")
        print("\n=== Testing (HR only) ===")

        # ── load weights ──────────────────────────────────────────
        if self.config.TOOLBOX_MODE == "only_test":
            ckpt = self.config.INFERENCE.MODEL_PATH
            if not os.path.exists(ckpt):
                raise ValueError(f"MODEL_PATH not found: {ckpt}")
            print("Loading pretrained model →", ckpt)
        else:  # train_and_test
            if self.config.TEST.USE_LAST_EPOCH:
                ckpt = os.path.join(
                    self.model_dir,
                    f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth"
                )
                print("Using last‑epoch model →", ckpt)
            else:
                ckpt = os.path.join(
                    self.model_dir,
                    f"{self.model_file_name}_Epoch{self.best_epoch}.pth"
                )
                print("Using best‑epoch model →", ckpt)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.model.eval().to(self.device)

        # ── storage dicts (HR only) ───────────────────────────────
        hr_predictions: dict = {}
        hr_labels:       dict = {}

        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for batch in tbar:
                vid, lbl = batch[0].to(self.device), batch[1].to(self.device)   # vid: [B,3,T,H,W] lbl: [B,2,T]
                subj_ids, sort_ids = batch[2], batch[3]                         # meta

                rppg_pred, _, _ = self.model(vid)                               # rppg_pred: [B,T]

                for i in range(vid.size(0)):
                    sid  = subj_ids[i]
                    idx  = int(sort_ids[i])

                    if sid not in hr_predictions:
                        hr_predictions[sid] = {}
                        hr_labels[sid]      = {}

                    hr_predictions[sid][idx] = rppg_pred[i].cpu()        # waveform
                    hr_labels[sid][idx]      = lbl[i, 0, :].cpu()        # GT HR channel

        # ── optional pickle dump ──────────────────────────────────
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(hr_predictions, hr_labels, self.config)

        # ── metric computation ────────────────────────────────────
        print("\nCalculating HR metrics …")
        calculate_metrics(hr_predictions, hr_labels, self.config)

        print("=== Done ===")


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
