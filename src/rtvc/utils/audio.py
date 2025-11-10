import numpy as np
import librosa
import soundfile as sf

def load_wav(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    wav = wav.astype(np.float32)
    return wav

def save_wav(path, wav, sr=22050):
    sf.write(path, wav, sr)

def trim_silence(wav, top_db=30):
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed

def mel_spectrogram(
    wav, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0, fmax=None
):
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    S = np.log(np.maximum(1e-5, S))
    return S
