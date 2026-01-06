"""Test script for efficiency metrics."""
import torch
from utils.metrics import count_encoder_parameters, compute_rtf, get_audio_duration_from_mel
from models.encoder import WhisperWrappedEncoder

# Create a simple config object
class Config:
    encoder_model = 'base'
    encoder_num_layers = None  # Test with all layers first

print('=== Loading Whisper Encoder ===')
encoder = WhisperWrappedEncoder.load(Config())

print('\n=== Parameter Counts (All 6 Layers) ===')
params_all = count_encoder_parameters(encoder, num_layers=None)
for k, v in params_all.items():
    if 'params' in k and isinstance(v, int):
        print(f'{k}: {v:,} ({v/1e6:.2f}M)')
    else:
        print(f'{k}: {v}')

print('\n=== Parameter Counts (4 Layers) ===')
params_4 = count_encoder_parameters(encoder, num_layers=4)
for k, v in params_4.items():
    if 'params' in k and isinstance(v, int):
        print(f'{k}: {v:,} ({v/1e6:.2f}M)')
    else:
        print(f'{k}: {v}')

print('\n=== Parameter Counts (2 Layers) ===')
params_2 = count_encoder_parameters(encoder, num_layers=2)
print(f"Used params: {params_2['used_params']:,} ({params_2['used_params']/1e6:.2f}M)")
print(f"Pruned params: {params_2['pruned_params']:,} ({params_2['pruned_params']/1e6:.2f}M)")

print('\n=== Layer Ablation Summary ===')
print('Layers | Used Params | % of Total')
print('-------|-------------|----------')
for n in range(1, 7):
    p = count_encoder_parameters(encoder, num_layers=n)
    pct = p['used_params'] / p['total_params'] * 100
    print(f'   {n}   | {p["used_params"]/1e6:.2f}M      | {pct:.1f}%')

print('\n=== RTF Test ===')
# Test RTF computation
rtf = compute_rtf(latency_ms=150, audio_duration_seconds=2.0)
print(f'Latency: 150ms, Audio: 2.0s -> RTF: {rtf:.4f}')
rtf2 = compute_rtf(latency_ms=500, audio_duration_seconds=1.0)
print(f'Latency: 500ms, Audio: 1.0s -> RTF: {rtf2:.4f}')

print('\n=== Audio Duration from Mel ===')
# Test with fake mel spectrogram (batch=1, n_mels=80, n_frames=300)
fake_mel = torch.randn(1, 80, 300)
duration = get_audio_duration_from_mel(fake_mel)
print(f'Mel shape {tuple(fake_mel.shape)} -> Duration: {duration:.2f}s')

# 3 seconds of audio (3000 frames at 10ms per frame)
fake_mel_3s = torch.randn(1, 80, 3000)  
duration_3s = get_audio_duration_from_mel(fake_mel_3s)
print(f'Mel shape {tuple(fake_mel_3s.shape)} -> Duration: {duration_3s:.2f}s')

print('\n=== All Efficiency Metrics Tests Passed! ===')
