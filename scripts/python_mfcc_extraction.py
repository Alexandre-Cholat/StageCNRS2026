from pyexpat import features
import numpy as np
import pandas as pd
import librosa
import numpy as np
import os


def extract_synchro_features_file(file_path):
    frame_size=0.025
    frame_stride=0.01
    num_ceps=13

    signal, fs = librosa.load(file_path, sr=None)

    # Frame parameters
    n_fft = int(frame_size * fs)
    hop_length = int(frame_stride * fs)

    # yin pitch estimator
    f0_values = librosa.yin(
        signal,
        fmin=50, fmax=500,
        sr=fs,
        frame_length=n_fft,
        hop_length=hop_length
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=fs,
        n_mfcc=num_ceps,
        n_fft=n_fft,
        hop_length=hop_length
    ).T  # shape: (num_frames, num_ceps)

    #Ensure alignment in t axis
    min_len = min(len(f0_values), len(mfcc))
    f0_values = f0_values[:min_len]
    mfcc = mfcc[:min_len, :]

    # Compute time vector
    time_values = librosa.frames_to_time(np.arange(min_len), sr=fs, hop_length=hop_length)

    # return Matrix = [ filename, frame_index, time, f0_value, 13 MFCCs ]
    rep_features = []
    filename = os.path.basename(file_path)
    
    for i in range(min_len):
        # Create a row with all the features
        row = [filename, i, time_values[i], np.log10(f0_values[i])]
        # Extend with MFCC coefficients
        row.extend(mfcc[i])
        rep_features.append(row)

    rep_features = np.array(rep_features, dtype=object)
    return rep_features





def extract_features_folder(folder_path):
    
    print("Elements in folder : ", len(os.listdir(folder_path)))
    loop_counter = 0

    file_name_f0_MFCC_matrix = []

    for filename in os.listdir(folder_path):
        loop_counter += 1
        # first 10 audio files
        if loop_counter >= len(os.listdir(folder_path)):
            break
        if filename.endswith('.wav'):

            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")


            # matrice de [  nom du fichier, numero du frame, log10(f0_value) , 13 mfcc coeffs ]
            features = extract_synchro_features_file(file_path)

            if features is not None:
                file_name_f0_MFCC_matrix.append(features)
            else:
                print(f"Warning: No features extracted for {filename}")

    
    return file_name_f0_MFCC_matrix
    

# main:

#single file extraction
column_names = ['filename', 'frame_index', 'time', 'log10(f0)'] + [f'mfcc_{i}' for i in range(13)]
features = extract_synchro_features_file("C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\LJSpeech-1.1\\LJSpeech-1.1\\big_wavs\\LJ001-0021.wav")
df = pd.DataFrame(features, columns=column_names)
df.to_csv('C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\python_MFCC_f0_extraction.csv', index=False)
print(f"Saved to C:\\Users\\alexa\\OneDrive\\Desktop\\Stage GIPSA-lab\\C++ audio-data_extraction\\python_MFCC_f0_extraction.csv")
