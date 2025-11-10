import numpy as np
import torch

class Synthesizer:
    """
    Tacotron-like interface. This stub maps (text, speaker_embedding) -> mel spectrogram.
    Replace the body with your Tacotron/Tacotron2 weights and text frontend.

    Expected method:
        synthesize_spectrograms(texts: list[str], embeds: np.ndarray) -> list[np.ndarray]
    """
    def __init__(self, model_ckpt=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Placeholder tiny net; for real use, load Tacotron2 + text cleaners.
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 80),  # 80 mel bins
        )
        self.eval = lambda: None
        if model_ckpt:
            self.proj.load_state_dict(torch.load(model_ckpt, map_location=self.device), strict=False)

    @torch.no_grad()
    def synthesize_spectrograms(self, texts, embeds):
        """
        texts: list[str]
        embeds: np.ndarray of shape [len(texts), 256]
        """
        # Placeholder: generate fixed-length mels conditioned on the speaker embedding.
        # Replace with true sequence-to-sequence Tacotron output.
        mel_list = []
        for i, t in enumerate(texts):
            e = torch.from_numpy(embeds[i]).float().unsqueeze(0)  # [1,256]
            base = self.proj(e)  # [1,80]
            # Tile to a short duration; in real model this length depends on text
            mel = base.repeat(1, 100).reshape(80, 100).cpu().numpy()  # [80, frames]
            mel_list.append(mel)
        return mel_list
