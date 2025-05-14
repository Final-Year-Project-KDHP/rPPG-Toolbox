#!/usr/bin/env python3

import os
import shutil
import re
import sys

# === User-editable section ===
# Paste your relative path here (it can contain line-breaks or spaces):
rel_path = """
runs/exp/PURE_SizeW128_SizeH128_ClipLength128_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_Ba
ckendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_PURE_PURE_PhysMambaSF_fc_sig_roun
d_mlti_shrdMLP_tasknorm_samedim_moe_pertsk_scalefuse0_3layer_all_config_oldfreqloss_harm_13_STFT0_Epoch49_PURE_outputs.pickle
"""
# Project root and destination folder (under root):
root = "/home/ddew0188/videopulse/rppg-toolbox/rppg-toolbox1multi"
dest_subfolder = "tools/output_signal_viz/picklecache"
# ==============================

# 1) Remove all whitespace (spaces, newlines, tabs) from the relative path\
clean_rel = re.sub(r"\s+", "", rel_path)

# 2) Build full source path and verify
src = os.path.join(root, clean_rel)
if not os.path.isfile(src):
    print(f"Error: Source file not found: {src}", file=sys.stderr)
    sys.exit(1)

# 3) Ensure destination directory exists

dest_folder = os.path.join(root, dest_subfolder)
os.makedirs(dest_folder, exist_ok=True)

# 4) Copy the pickle file

dest = os.path.join(dest_folder, os.path.basename(clean_rel))
shutil.copy2(src, dest)

print(f"Copied:\n  {src}\nto\n  {dest}")
