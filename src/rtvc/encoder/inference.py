import torch
import torchaudio
import numpy as np
from ..utils.audio import load_wav, trim_silence

class SpeakerEncoder(torch.nn.Module):
    """
    Interface-compatible placeholder.
    Swap in your trained GE2E model by loading its state_dict into this module
    or replacing this file with the original implementation.
    """
    def __init__(self, model_ckpt=None, device=None, sample_rate=16000):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Tiny example network; replace with real GE2E encoder.
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.proj = torch.nn.Linear(128, 256)
        self.sample_rate = sample_rate
        self.eval()

        if model_ckpt:
            self.load_state_dict(torch.load(model_ckpt, map_location=self.device), strict=False)

    @torch.no_grad()
    def embed_utterance(self, wav_np: np.ndarray):
        wav = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [B,1,T]
        h = self.backbone(wav).squeeze(-1)  # [B,128]
        e = torch.nn.functional.normalize(self.proj(h), dim=-1)  # [B,256]
        return e.squeeze(0).cpu().numpy()

    def embed_from_file(self, path: str):
        wav = load_wav(path, sr=self.sample_rate)
        wav = trim_silence(wav)
        return self.embed_utterance(wav)
