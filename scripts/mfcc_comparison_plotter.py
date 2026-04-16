import pandas as pd
import matplotlib.pyplot as plt

def plot_mfcc_comparison(csv_python, csv_cpp, mfcc_cols=['mfcc_0', 'mfcc_1', 'mfcc_2']):
    """
    Plots a comparison of MFCC curves from two CSV files for a specific audio file.
    """
    # Load the datasets
    df_py = pd.read_csv(csv_python)
    df_cpp = pd.read_csv(csv_cpp)

    df_py_filtered = df_py
    df_cpp_filtered = df_cpp

    # Set up the subplots
    num_plots = len(mfcc_cols)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    
    # Handle the case where we only plot 1 coefficient
    if num_plots == 1:
        axes = [axes]

    # Plot each MFCC coefficient
    for ax, col in zip(axes, mfcc_cols):
        ax.plot(df_py_filtered['frame_index'], df_py_filtered[col], 
                label=f'Python {col}', linewidth=2, alpha=0.8)
        
        ax.plot(df_cpp_filtered['frame_index'], df_cpp_filtered[col], 
                label=f'C++ {col}', linewidth=2, linestyle='--', alpha=0.8)
        
        ax.set_title(f'Comparison of {col}')
        ax.set_ylabel('Magnitude')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.xlabel('Frame Index')
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import idct
from scipy.stats import spearmanr

# 1. Added target_wav and mel_cols as proper arguments
def plot_mel_reconstruction(csv_python, csv_cpp, target_wav, mel_cols=['mel_0', 'mel_1', 'mel_2']):
    """
    Plots a comparison of reconstructed Log-Mel curves from two CSV files.
    """
    # Load the datasets
    df_py = pd.read_csv(csv_python)
    df_cpp = pd.read_csv(csv_cpp)

    # Reconstruct Mel-Spectrum via iDCT for all frames
    mfcc_cols = [f'mfcc_{i}' for i in range(13)]
    all_mel_cols = [f'mel_{i}' for i in range(26)]
    
    # Apply iDCT to the 13 MFCCs to get 26 Mel bins, matching librosa's ortho norm
    df_py[all_mel_cols] = idct(df_py[mfcc_cols].values, axis=1, n=26, norm='ortho')
    df_cpp[all_mel_cols] = idct(df_cpp[mfcc_cols].values, axis=1, n=26, norm='ortho')

    # 2. Restored the filtering structure to isolate the specific WAV file
    df_py_filtered = df_py[df_py['filename'] == target_wav]
    df_cpp_filtered = df_cpp[df_cpp['filename'] == target_wav]

    # Set up the subplots
    num_plots = len(mel_cols)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    
    # Handle the case where we only plot 1 coefficient
    if num_plots == 1:
        axes = [axes]

    # Plot each reconstructed Mel bin
    for ax, col in zip(axes, mel_cols):
        ax.plot(df_py_filtered['frame_index'], df_py_filtered[col], 
                label=f'Python {col}', linewidth=2, alpha=0.8)
        
        ax.plot(df_cpp_filtered['frame_index'], df_cpp_filtered[col], 
                label=f'C++ {col}', linewidth=2, linestyle='--', alpha=0.8)
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(df_py_filtered[col], df_cpp_filtered[col])
        
        ax.set_title(f'Comparison of {col} (Spearman r = {corr:.4f})')
        ax.set_ylabel('Log-Energy')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.xlabel('Frame Index')
    plt.tight_layout()
    plt.show()

# --- Execution ---
# Note: Added 'r' before the string to handle Windows backslashes properly
python_csv_path = r'C:\Users\alexa\OneDrive\Desktop\Stage GIPSA-lab\C++ audio-data_extraction\python_mfcc_extraction.csv' 
cpp_csv_path = r'C:\Users\alexa\OneDrive\Desktop\Stage GIPSA-lab\C++ audio-data_extraction\cpp_mfcc_extraction-corrected.csv'
wav_to_check = 'LJ001-0021.wav'

print(f"Plotting comparison for {wav_to_check}...")

# 3. Pass the file, the wav name, and a list of specific Mel bins to plot
plot_mel_reconstruction(python_csv_path, cpp_csv_path, target_wav=wav_to_check, mel_cols=['mel_0', 'mel_5', 'mel_15'])

# --- Execution ---
# Replace these with your actual file paths
# python_csv_path = 'C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\python_mfcc_extraction.csv' 
# cpp_csv_path = 'C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\cpp_mfcc_extraction.csv'
# wav_to_check = 'LJ001-0021.wav'

#print(f"Plotting comparison for {wav_to_check}...")

#plot_mfcc_comparison(python_csv_path, cpp_csv_path, wav_to_check, ['mfcc_0', 'mfcc_1', 'mfcc_2'])

