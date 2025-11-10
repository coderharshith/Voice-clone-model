import os
import click
import numpy as np
from src.rtvc.encoder.inference import SpeakerEncoder
from src.rtvc.synthesizer.inference import Synthesizer
from src.rtvc.vocoder.inference import Vocoder
from src.rtvc.utils.audio import load_wav, save_wav, trim_silence

@click.command()
@click.option("--reference", "-r", type=click.Path(exists=True), required=True, help="Reference voice WAV (a few seconds).")
@click.option("--text", "-t", type=str, required=True, help="Text to synthesize.")
@click.option("--out", "-o", type=str, default="cloned.wav", help="Output WAV path.")
@click.option("--encoder-ckpt", type=click.Path(exists=True), default=None, help="(Optional) GE2E encoder checkpoint.")
@click.option("--synth-ckpt", type=click.Path(exists=True), default=None, help="(Optional) Tacotron checkpoint.")
@click.option("--sample-rate", type=int, default=22050)
def main(reference, text, out, encoder_ckpt, synth_ckpt, sample_rate):
    print("Loading modules...")
    encoder = SpeakerEncoder(model_ckpt=encoder_ckpt, sample_rate=16000)
    synth = Synthesizer(model_ckpt=synth_ckpt)
    vocoder = Vocoder(sample_rate=sample_rate)

    print("Embedding speaker...")
    ref_wav = load_wav(reference, sr=16000)
    ref_wav = trim_silence(ref_wav)
    embed = encoder.embed_utterance(ref_wav)  # [256]

    print("Synthesizing mel...")
    mels = synth.synthesize_spectrograms([text], np.stack([embed]))  # list of [80, T]
    mel = mels[0]

    print("Vocoder inference...")
    wav = vocoder.infer_waveform(mel)

    print(f"Saving: {out}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_wav(out, wav, sr=sample_rate)
    print("Done.")

if __name__ == "__main__":
    main()
