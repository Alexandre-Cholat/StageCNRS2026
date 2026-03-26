import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_comparison(cpp_csv_path, pytorch_csv_path, num_frames):
    """
    Reads predictions from C++ and PyTorch and superimposes them.
    start_frame / num_frames: Use these to zoom in on a specific sentence.
    """
    try:
        df_cpp = pd.read_csv(cpp_csv_path)
        df_pyt = pd.read_csv(pytorch_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        sys.exit(1)

    # first N rows
    df_cpp_slice = df_cpp.iloc[:num_frames]
    df_pyt_slice = df_pyt.iloc[:num_frames]

    # Extract values
    target_f0 = df_cpp_slice['target_f0'].values
    cpp_preds = df_cpp_slice['f0_pred'].values
    pyt_preds = df_pyt_slice['f0_pred'].values

    # Create a continuous sequential X-axis (0 to num_frames-1)
    # This prevents the plot from looping back on itself when frame_idx resets
    x_axis = np.arange(len(target_f0))

    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot Ground Truth
    plt.plot(x_axis, target_f0, label="Target F0", color='black', linewidth=2.5, linestyle='--')
    
    # Plot PyTorch (Offline)
    plt.plot(x_axis, pyt_preds, label="PyTorch pred.", color='blue', alpha=0.7, linewidth=2)
    
    # Plot C++ (Real-Time)
    plt.plot(x_axis, cpp_preds, label="RTNeural pred.", color='red', alpha=0.7, linewidth=2)

    # Aesthetics
    plt.title(f"Offline vs Real-Time f0_pred - {num_frames} frames", fontsize=16)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("log10 F0", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    
    # Add minor gridlines for easier visual alignment
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.minorticks_on()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update these with your exact generated filenames
    CPP_CSV = "C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\audio-data-prediction\\real-time-corrected-predictions_2026-03-26_14-50-57.csv"
    PYTORCH_CSV = "C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\audio-data-prediction\\pytorch-offline-preds_2026-03-26_13-57-40.csv"
    
    # Adjust start_frame and num_frames to isolate exactly one sentence
    visualize_comparison(CPP_CSV, PYTORCH_CSV, num_frames=200)