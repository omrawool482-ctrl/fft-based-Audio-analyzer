
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks

FS = 16000      # sampling rate (Hz)
N_FFT = 512     # frame size for the spectrogram
HOP = 256       # step between frames

# ---------------------------------------------------------------- 1. find the music folder
here = os.path.dirname(os.path.abspath(__file__))
candidates = [os.path.join(here, "..", "dataset", "music"),
              os.path.join(here, "dataset", "music"),
              os.path.join(here, "music")]
folder = next((p for p in candidates if os.path.isdir(p)), None)
if folder is None:
    raise SystemExit("music folder not found. Put .wav files in dataset/music next to this file.")
files = sorted(glob.glob(os.path.join(folder, "*.wav")))
print("music folder:", os.path.abspath(folder), "| clips:", len(files))

# ---------------------------------------------------------------- 2. helper functions
def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)                      # stereo -> mono
    return x / (np.max(np.abs(x)) + 1e-12)      # scale to -1..1

def spectrum(x):
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))      # FFT of the whole clip
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return freqs, mag

def features(x):
    freqs, mag = spectrum(x)
    power = mag ** 2
    return {
        "dominant_hz": freqs[np.argmax(mag)],                                # strongest frequency
        "centroid_hz": np.sum(freqs * power) / np.sum(power),                # "brightness"
        "low_energy_%": 100 * power[freqs < 1000].sum() / power.sum(),       # energy below 1 kHz
        "zcr": np.mean(np.abs(np.diff(np.sign(x))) > 0),                     # zero-crossing rate
    }

def spectrogram(x):
    n_frames = 1 + (len(x) - N_FFT) // HOP
    win = np.hanning(N_FFT)
    S = np.array([np.abs(np.fft.rfft(x[i * HOP:i * HOP + N_FFT] * win)) for i in range(n_frames)])
    return S                                                                   # shape: frames x bins

def detect_events(S):
    """Note onsets: spectral flux = how much the spectrum increases between frames."""
    L = np.log1p(S)
    flux = np.r_[0, np.maximum(np.diff(L, axis=0), 0).sum(axis=1)]
    threshold = flux.mean() + 1.5 * flux.std()
    peaks, _ = find_peaks(flux, height=threshold, distance=int(0.15 * FS / HOP))
    return peaks * HOP / FS, flux, threshold                                   # onset times in seconds

# ---------------------------------------------------------------- 3. features for all clips
print("\n%-18s %12s %12s %14s %8s" % ("clip", "dominant Hz", "centroid Hz", "low energy %", "ZCR"))
for f in files:                                    # all clips
    ft = features(load(f))
    print("%-18s %12.0f %12.0f %14.1f %8.3f" % (os.path.basename(f), ft["dominant_hz"],
          ft["centroid_hz"], ft["low_energy_%"], ft["zcr"]))

# ---------------------------------------------------------------- 4. spectrum of EACH clip
out_dir = os.path.join(here, "spectra")
os.makedirs(out_dir, exist_ok=True)
n = len(files)
cols = 5
rows = int(np.ceil(n / cols))
fig_all, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True)
axs = np.atleast_1d(axs).ravel()
for ax, f in zip(axs, files):
    name = os.path.splitext(os.path.basename(f))[0]
    freqs, mag = spectrum(load(f))
    db = 20 * np.log10(mag + 1e-9)
    ax.plot(freqs, db, lw=0.5)                              # small plot in the grid
    ax.set_title(name, fontsize=9)
    fig, a = plt.subplots(figsize=(8, 3.5))                 # separate image for this clip
    a.plot(freqs, db, lw=0.6)
    a.set_title(name + " - FFT magnitude spectrum"); a.set_xlabel("frequency (Hz)"); a.set_ylabel("dB")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, name + ".png"), dpi=110); plt.close(fig)
for ax in axs[n:]:
    ax.axis("off")
fig_all.supxlabel("frequency (Hz)"); fig_all.supylabel("dB")
fig_all.tight_layout(); fig_all.savefig(os.path.join(here, "all_spectra.png"), dpi=110)
print("\nsaved: %d spectrum images in %s and all_spectra.png" % (n, out_dir))

# ---------------------------------------------------------------- 5. detailed analysis of EACH clip
an_dir = os.path.join(here, "analysis")
os.makedirs(an_dir, exist_ok=True)
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    x = load(f)
    t = np.arange(len(x)) / FS
    freqs, mag = spectrum(x)
    S = spectrogram(x)
    onsets, flux, thr = detect_events(S)
    print(name, "| note events at (s):", np.round(onsets, 2))

    fig, ax = plt.subplots(4, 1, figsize=(11, 11))
    ax[0].plot(t, x, lw=0.6)
    ax[0].set_title(name + " - Waveform"); ax[0].set_xlabel("time (s)")
    ax[1].plot(freqs, 20 * np.log10(mag + 1e-9), lw=0.6)
    ax[1].set_title("FFT magnitude spectrum"); ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("dB")
    ax[2].imshow(20 * np.log10(S.T + 1e-6), origin="lower", aspect="auto", cmap="magma",
                 extent=[0, len(x) / FS, 0, FS / 2])
    for o in onsets:
        ax[2].axvline(o, color="lime", lw=1)
    ax[2].set_title("Spectrogram with detected note events (green)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("Hz")
    ax[3].plot(np.arange(len(flux)) * HOP / FS, flux, label="spectral flux")
    ax[3].axhline(thr, color="r", ls="--", label="threshold")
    ax[3].set_title("Spectral flux (event detector)"); ax[3].set_xlabel("time (s)"); ax[3].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(an_dir, name + "_analysis.png"), dpi=100)
    plt.close(fig)                                   # close so 20 windows do not pile up
print("saved %d analysis images in %s" % (len(files), an_dir))