import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.fftpack  # <--- NOUVEL IMPORT NÉCESSAIRE

# 1. Paramètres et chargement
wav_path = r"C:\Users\alexa\OneDrive\Desktop\Stage GIPSA-lab\LJSpeech-1.1\LJSpeech-1.1\big_wavs\LJ001-0021.wav"
y, sr = librosa.load(wav_path, sr=None)

n_fft = 2048
n_mels = 128
n_mfcc = 13

# 2. Extraction d'une trame spécifique (ex: 1.5 secondes)
time_sec = 2.1
frame_center = int(time_sec * sr)
start = max(0, frame_center - n_fft // 2)
frame = y[start:start + n_fft]

# Fenêtrage
window = np.hanning(len(frame))
frame_windowed = frame * window

# 3. Calcul du spectre FFT brut et de l'enveloppe Mel
power_spec = np.abs(np.fft.rfft(frame_windowed)) ** 2
power_db = librosa.power_to_db(power_spec, ref=np.max)
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
mel_energies = np.dot(mel_basis, power_spec)
mel_db = librosa.power_to_db(mel_energies, ref=np.max)

mel_freqs_edges = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=sr/2.0)
mel_center_freqs = mel_freqs_edges[1:-1]

# --- CORRECTION : Calcul des MFCC via SciPy ---
# On calcule la DCT sur les 128 bandes Mel (type 2, norm='ortho' est le standard Librosa)
mfccs_complets = scipy.fftpack.dct(mel_db, type=2, norm='ortho')

# On ne garde que les 13 premiers coefficients, comme d'habitude
mfccs = mfccs_complets[:n_mfcc]

# 4. Affichage Graphique
plt.figure(figsize=(14, 8))

# Le spectre brut en arrière-plan très clair
plt.plot(freqs, power_db, color='lightgray', label='Spectre FFT brut', linewidth=1, alpha=0.5)

# L'enveloppe Mel complète (128 bandes) en pointillé noir pour servir de cible
spl_cible = make_interp_spline(mel_center_freqs, mel_db, k=3)
freqs_smooth = np.linspace(mel_center_freqs.min(), mel_center_freqs.max(), 500)
plt.plot(freqs_smooth, spl_cible(freqs_smooth), color='black', linestyle='--', label='Enveloppe Cumulative (128 bandes Mel)', linewidth=2.5)

colors = plt.cm.magma(np.linspace(0.2, 0.9, 8))

# 5. Boucle de reconstruction cumulative (De C1 à C7)
for i in range(1, 6):
    # Pour reconstruire l'enveloppe, on a besoin d'un tableau de 128 valeurs (n_mels)
    mfccs_partiels = np.zeros(n_mels)
    
    # On insère C0 jusqu'à Ci, et on laisse le reste à zéro
    mfccs_partiels[0:i+1] = mfccs[0:i+1]
    
    # Transformée Inverse (IDCT) : Des MFCC vers le spectre Mel en dB
    mel_reconstruit_db = scipy.fftpack.idct(mfccs_partiels, type=2, norm='ortho')
    
    # Lissage spline pour l'affichage
    spl_recon = make_interp_spline(mel_center_freqs, mel_reconstruit_db, k=3)
    
    
    if i == 1:
        label = "Enveloppe (MFCC 0 à 1)"
    elif i == 2:
        label = "Enveloppe (MFCC 0 à 2)"
    else:
        label = f"Enveloppe (MFCC 0 à {i})"
        
    plt.plot(freqs_smooth, spl_recon(freqs_smooth), color=colors[i], label=label, linewidth=1 + (i*0.2))



# Finitions
plt.xlim(0, 8000)
plt.ylim(-80, 5)
plt.title(f"Superposition d'une trame FFT et de la reconstruction cumulative de son Enveloppe Mel", fontsize=14)
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()