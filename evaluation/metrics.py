import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
from scipy.stats import beta
import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.ndimage import convolve1d

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
    
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        print("Ground Truths:", list(gt_hr_fft_all))
        print("Predicted Heart Rates:", list(predict_hr_fft_all))
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                MSE_FFT = np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all))
                RMSE_FFT = np.sqrt(MSE_FFT)
                # In the case of standard error (SE) for RMSE, scale the standard error relative to the RMSE
                # which is less sensitive to possible outliers. This should prevent the SE from becoming too
                # large, and exaggerated, due to large, squared errors.
                MSE_FFT_se = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                standard_error = MSE_FFT_se / (2 * np.sqrt(MSE_FFT))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                # compare.scatter_plot(
                #     x_label='GT PPG HR [bpm]',
                #     y_label='rPPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                #     file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                # compare.difference_plot(
                #     x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                #     y_label='Average of rPPG HR and GT PPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                #     file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        print("Ground Truths:", list(gt_hr_peak_all))
        print("Predicted Heart Rates:", list(predict_hr_peak_all))
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                MSE_PEAK = np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all))
                RMSE_PEAK = np.sqrt(MSE_PEAK)
                # In the case of standard error (SE) for RMSE, scale the standard error relative to the RMSE
                # which is less sensitive to possible outliers. This should prevent the SE from becoming too
                # large, and exaggerated, due to large, squared errors.
                MSE_PEAK_se = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                standard_error = MSE_PEAK_se / (2 * np.sqrt(MSE_PEAK))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                # compare.scatter_plot(
                #     x_label='GT PPG HR [bpm]',
                #     y_label='rPPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                #     file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                # compare.difference_plot(
                #     x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                #     y_label='Average of rPPG HR and GT PPG HR [bpm]',
                #     show_legend=True, figure_size=(5, 5),
                #     the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                #     file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
    


def plot_label_distribution(data_loader, save_dir, filename="label_distribution.png", bin_data_file="label_bins.npy"):
    labels = []
    for batch in data_loader:
        # Expecting: (data, label_tensor, [optional filename])
        label_batch = batch[1]  # shape: [B, 2, T]
        label_values = label_batch[:, 1:2, :].mean(dim=(1, 2)).cpu().numpy()
        labels.extend(label_values.tolist())

    # Filter out values below 90
    labels = [label for label in labels if label >= 90]

    # Define bins for SpO₂ values from 90 to 100 inclusive
    bins = np.arange(89.5, 101.5, 1)
    y, x = np.histogram(labels, bins=bins)

    # Bin centers for easier interpretation and plotting
    bin_centers = (x[:-1] + x[1:]) / 2

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, y, width=1.0, align='center')
    plt.title("Mean SpO₂ Label Distribution per Video (≥ 90)")
    plt.xlabel("Mean SpO₂ Value")
    plt.ylabel("Video Count")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    # Save bin data (counts and bin centers) to file for later analysis
    bin_data = {"bin_centers": bin_centers, "counts": y}
    np.save(os.path.join(save_dir, bin_data_file), bin_data)

    # Return for use in code (optional)
    return bin_centers, y

def get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2):
    assert kernel == 'gaussian', "Only gaussian supported"
    half_ks = (ks - 1) // 2
    base = np.arange(-half_ks, half_ks + 1)
    kernel = np.exp(-0.5 * (base / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def get_exponential_kernel(ks=5, decay=1.0, direction="right"):
    base = np.arange(ks)
    if direction == "left":
        base = base[::-1]
    kernel = np.exp(-base / decay)
    kernel /= kernel.sum()
    return kernel



def get_beta_kernel(ks=15, a=2, b=5):
    x = np.linspace(0, 1, ks)
    kernel = beta.pdf(x, a, b)
    kernel /= kernel.sum()
    return kernel


def compute_lds_weights_with_plots(mean_labels, save_dir, num_bins=50, ks=7, a=2, b=5):
    os.makedirs(save_dir, exist_ok=True)

    # Filter labels >= 90
    filtered_labels = [x for x in mean_labels if x >= 90]

    # Define bin edges and digitize
    bin_edges = np.linspace(90, 100, num_bins + 1)
    bin_indices = np.digitize(filtered_labels, bin_edges) - 1

    # Empirical distribution
    count_per_bin = dict(Counter(bin_indices))
    emp_dist = [count_per_bin.get(i, 0) for i in range(num_bins)]

    # Apply smoothing
    lds_kernel = get_beta_kernel(ks=ks, a=a, b=b)
    eff_dist = convolve1d(np.array(emp_dist), weights=lds_kernel, mode='constant')

    # Compute bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot empirical distribution
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, emp_dist, width=(bin_edges[1] - bin_edges[0]), color='skyblue')
    plt.title("Original Label Distribution")
    plt.xlabel("Mean SpO₂")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "label_distribution_original.png"))
    plt.close()

    # Plot smoothed distribution
    plt.figure(figsize=(10, 5))
    plt.bar(bin_centers, eff_dist, width=(bin_edges[1] - bin_edges[0]), color='salmon')
    plt.title("Smoothed Label Distribution (Beta Kernel)")
    plt.xlabel("Mean SpO₂")
    plt.ylabel("Smoothed Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "label_distribution_smoothed.png"))
    plt.close()

    # Compute weights for all labels
    weights = []
    for label in mean_labels:
        if label < 90:
            weights.append(0.0)
        else:
            bin_idx = np.digitize(label, bin_edges) - 1
            eff_freq = eff_dist[bin_idx]
            weights.append(float(1.0 / (eff_freq + 1e-6)))

    return weights