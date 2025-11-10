import numpy as np
import librosa

class Vocoder:
    """
    WaveRNN/WaveGlow/HiFi-GAN interface. This fallback uses Griffin-Lim for a CPU-only demo.
    Replace with a real neural vocoder by loading weights and calling its inference.
    """
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=None):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.win = win_length
        self.fmin = fmin
        self.fmax = fmax

    def infer_waveform(self, mel_log):
        # invert log-mel to magnitude, then Griffin-Lim back to waveform
        mel = np.exp(mel_log)
        inv_mel = librosa.feature.inverse.mel_to_stft(
            mel, sr=self.sr, n_fft=self.n_fft, fmin=self.fmin, fmax=self.fmax
        )
        wav = librosa.griffinlim(
            inv_mel, hop_length=self.hop, win_length=self.win, n_iter=60
        )
        return wav.astype(np.float32)
