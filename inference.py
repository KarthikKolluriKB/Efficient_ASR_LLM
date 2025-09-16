import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np 

from transformers import AutoProcessor
from models.model import model_builder 
from omegaconf import OmegaConf


def preprocess_audio(audio_path: str, target_sr: int = 16000):
    """
    Returns:
      audio_16k: np.ndarray float32, shape [T]
      sr: int (target_sr)
    """
    # Load: waveform [C, T], sample_rate: int
    waveform, sr = torchaudio.load(audio_path)

    # Downmix to mono if multi-channel
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Resample if needed
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr
 
    return waveform.squeeze(0).contiguous().float().numpy(), sr # [T]

def load_audio_and_mel(audio_path, feature_extractor, device , n_mels: int = 80):
    """
    Loads audio file and returns Whisper-style mel spectogram [1, n_mel, T].
    """
    
    wav, sr = preprocess_audio(audio_path=audio_path)

    # Feature extractor with Hugging Face extractor
    result = feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
    mel = result["input_features"] # [1, n_mels, T]
    return mel.to(device)

def main():
    #parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    # parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    # parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")
    # parser.add_argument("--prompt", type=str, default="<|ASR|>", help="(optional) Conditioning Prompt")
    # parser.add_argument("--max_new_tokens", type=int, default=64)
    #args = parser.parse_args()

    args = argparse.Namespace(config='configs/config.yaml', 
                              ckpt_path="outputs_3b_100h\projector_best_epoch_final.pt",
                              prompt="Transcribe speech to text.",
                              audio_path="audio/test.flac",
                              max_new_tokens=64)

    # Load config and build model/tokenizer
    cfg = OmegaConf.load(args.config)
    model, tokenizer = model_builder(cfg.train, cfg.model, ckpt_path=args.ckpt_path)
    # Setting model to eval
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load Whisper feature extractor
    feature_extractor = AutoProcessor.from_pretrained(cfg.model.encoder_model)

    # Prepare and run inference 
    mel = load_audio_and_mel(args.audio_path, feature_extractor, device, n_mels=cfg.data.mel_size)

    # permuting 
    mel = mel.permute((0,2,1))

    text = model.inference(
        audio_mel=mel, 
        prompt=args.prompt, 
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    print(f"Transcription: {text}")

if __name__ == "__main__":
    main()


#Example usage: 

# python inference.py --config configs/test_config.yaml --ckpt_path outputs_smoke_1b/projector_best.pt --audio_path path/to/file.wav
