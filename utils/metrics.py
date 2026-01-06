from typing import List, Tuple, Union, Dict
import torch
import numpy as np
import re
import unicodedata
from transformers import AutoTokenizer
import jiwer  # Direct jiwer - same as HuggingFace evaluate uses internally


class DanishTextNormalizer:
    """
    Text normalizer for Danish ASR evaluation.
    
    Based on OpenAI Whisper's BasicTextNormalizer, adapted for Danish.
    Key difference: Preserves Danish special characters (æ, ø, å) which are
    distinct letters in the Danish alphabet, not diacritics.
    
    Reference: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
    """
    
    def __call__(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        
        # Remove content in brackets [like this] or <like this>
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        
        # Remove content in parentheses (like this)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        
        # Normalize unicode (NFKC keeps æ, ø, å intact)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove punctuation and symbols, but keep letters (including æøå) and numbers
        # Category M = Mark, S = Symbol, P = Punctuation
        text = "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c
            for c in text
        )
        
        # Collapse multiple spaces into single space
        text = re.sub(r"\s+", " ", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# Global normalizer instance
_danish_normalizer = DanishTextNormalizer()


def normalize_danish(text: str) -> str:
    """Apply Danish ASR text normalization."""
    return _danish_normalizer(text)


# jiwer transformation using our Danish normalizer
# Compatible with jiwer 3.x API using process_words
DANISH_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",  # Remove content in brackets
        r"\(([^)]+?)\)": "",       # Remove content in parentheses
    }),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])

def compute_accuracy(pad_outputs: torch.LongTensor,
                     pad_targets: torch.LongTensor,
                     ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).
    """

    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()

def decode_texts_from_outputs(logits: torch.Tensor,
                            labels: torch.Tensor, 
                            tokenizer: AutoTokenizer,
                            ignore_label: int = -100) -> Tuple[List[str], List[str]]:
    """Decode model outputs and labels into texts.

    Args:
        logits (torch.Tensor): Prediction logits (B, L, V).
        labels (torch.Tensor): Target label tensors (B, L).
        tokenizer (AutoTokenizer): Tokenizer for decoding indices to text.
        ignore_label (int): Label to ignore in decoding.

    Returns:
        Tuple[List[str], List[str]]: Tuple of (hypothesis texts, reference texts).
    """
    # Get predictions by taking argmax of logits
    pred_ids = torch.argmax(logits, dim=-1)  # (B, L)
    
    # Align sequence lengths
    seq_diff = pred_ids.size(1) - labels.size(1)
    if seq_diff: 
        raise ValueError(f"Prediction and label sequence lengths do not match: {pred_ids.size(1)} vs {labels.size(1)}")
        
    # Create mask for valid tokens
    valid_mask = (labels != ignore_label)
    
    hyp_texts, ref_texts = [], []
    for pred, label, mask in zip(pred_ids, labels, valid_mask):
        # Select only valid tokens
        valid_pred = pred[mask]
        valid_label = label[mask]
        
        if len(valid_pred) == 0 or len(valid_label) == 0:
            continue
            
        try:
            pred_text = tokenizer.decode(valid_pred, skip_special_tokens=True).strip()
            label_text = tokenizer.decode(valid_label, skip_special_tokens=True).strip()
            
            if pred_text and label_text:
                hyp_texts.append(pred_text)
                ref_texts.append(label_text)
        except:
            raise ValueError("Decoding failed for predictions or labels.")
            
    return hyp_texts, ref_texts


def compute_wer(hyp_texts: List[str], ref_texts: List[str], normalize: bool = True) -> float:
    """Calculate Word Error Rate (WER) from hypothesis and reference texts.
    
    Uses jiwer library - the standard WER implementation used by most ASR papers.
    Applies Danish-specific text normalization (preserves æ, ø, å).
    
    Based on OpenAI Whisper's BasicTextNormalizer for non-English languages.
    
    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts
        ref_texts (List[str]): List of reference (ground truth) texts
        normalize (bool): Whether to apply Danish ASR normalization 
                         (lowercase, remove punctuation, preserve æøå).
                         Default True for fair comparison with published results.
        
    Returns:
        float: Word Error Rate score (0.0 = perfect, 1.0 = 100% error)
    """
    if not hyp_texts or not ref_texts:
        return 0.0
    
    # Compute WER with Danish normalization using process_words API
    if normalize:
        output = jiwer.process_words(
            ref_texts, 
            hyp_texts,
            reference_transform=DANISH_TRANSFORM,
            hypothesis_transform=DANISH_TRANSFORM
        )
        return output.wer
    else:
        # Raw WER without normalization
        return jiwer.wer(ref_texts, hyp_texts)


# =============================================================================
# CHARACTER ERROR RATE (CER)
# =============================================================================

def compute_cer(hyp_texts: List[str], ref_texts: List[str], normalize: bool = True) -> float:
    """Calculate Character Error Rate (CER) from hypothesis and reference texts.
    
    CER is particularly useful for Danish due to compound words 
    (e.g., "arbejdsløshedsforsikring" = unemployment insurance).
    A small spelling error affects WER heavily but CER more fairly.
    
    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts
        ref_texts (List[str]): List of reference (ground truth) texts
        normalize (bool): Whether to apply Danish ASR normalization.
        
    Returns:
        float: Character Error Rate score (0.0 = perfect, 1.0 = 100% error)
    """
    if not hyp_texts or not ref_texts:
        return 0.0
    
    if normalize:
        # Apply Danish normalization before CER computation
        normalized_hyp = [normalize_danish(text) for text in hyp_texts]
        normalized_ref = [normalize_danish(text) for text in ref_texts]
        return jiwer.cer(normalized_ref, normalized_hyp)
    else:
        return jiwer.cer(ref_texts, hyp_texts)


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

def count_encoder_parameters(encoder, num_layers: int = None) -> Dict[str, int]:
    """
    Count parameters in the Whisper encoder, considering layer pruning.
    
    Args:
        encoder: Whisper encoder module
        num_layers: Number of layers being used (None = all layers)
        
    Returns:
        Dict with parameter counts:
            - total_params: All encoder parameters
            - used_params: Parameters in layers actually used
            - pruned_params: Parameters in pruned (unused) layers
            - conv_params: Parameters in conv layers (always used)
            - block_params_per_layer: Parameters per transformer block
    """
    # Conv layers (always used regardless of pruning)
    conv_params = sum(p.numel() for p in encoder.conv1.parameters())
    conv_params += sum(p.numel() for p in encoder.conv2.parameters())
    
    # Positional embedding
    pos_params = encoder.positional_embedding.numel()
    
    # Layer norm (always used)
    ln_params = sum(p.numel() for p in encoder.ln_post.parameters())
    
    # Transformer blocks
    total_blocks = len(encoder.blocks)
    block_params_list = [sum(p.numel() for p in block.parameters()) for block in encoder.blocks]
    total_block_params = sum(block_params_list)
    
    # Calculate used vs pruned
    if num_layers is None:
        num_layers = total_blocks
    
    used_block_params = sum(block_params_list[:num_layers])
    pruned_block_params = sum(block_params_list[num_layers:])
    
    # Total calculations
    total_params = conv_params + pos_params + ln_params + total_block_params
    used_params = conv_params + pos_params + ln_params + used_block_params
    
    return {
        "total_params": total_params,
        "used_params": used_params,
        "pruned_params": total_params - used_params,
        "conv_params": conv_params,
        "pos_embedding_params": pos_params,
        "ln_post_params": ln_params,
        "block_params_per_layer": block_params_list[0] if block_params_list else 0,
        "num_layers_used": num_layers,
        "num_layers_total": total_blocks,
    }


def measure_inference_latency(
    model,
    sample_input: Dict[str, torch.Tensor],
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """
    Measure inference latency with proper GPU timing.
    
    Uses CUDA events for accurate GPU timing, with warmup runs
    to ensure GPU is at peak performance.
    
    Args:
        model: The ASRLLM model
        sample_input: Dict with model inputs (input_ids, attention_mask, audio_mel, etc.)
        device: Device to run on ("cuda" or "cpu")
        num_warmup: Number of warmup runs (not measured)
        num_runs: Number of timed runs for averaging
        
    Returns:
        Dict with timing statistics:
            - mean_latency_ms: Average latency in milliseconds
            - std_latency_ms: Standard deviation
            - min_latency_ms: Minimum latency
            - max_latency_ms: Maximum latency
    """
    model.eval()
    
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in sample_input.items()}
    
    latencies = []
    
    with torch.no_grad():
        # Warmup runs (not measured) - critical for accurate GPU timing
        for _ in range(num_warmup):
            _ = model(**inputs)
        
        # Synchronize before starting measurements
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Timed runs
        for _ in range(num_runs):
            if device == "cuda":
                # Use CUDA events for precise GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(**inputs)
                end_event.record()
                
                # Wait for completion
                torch.cuda.synchronize()
                
                # Get elapsed time in milliseconds
                latency_ms = start_event.elapsed_time(end_event)
            else:
                # CPU timing
                import time
                start = time.perf_counter()
                _ = model(**inputs)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
            
            latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "num_runs": num_runs,
    }


def compute_rtf(
    latency_ms: float,
    audio_duration_seconds: float,
) -> float:
    """
    Compute Real-Time Factor (RTF).
    
    RTF = processing_time / audio_duration
    - RTF < 1.0 means faster than real-time (good!)
    - RTF = 1.0 means exactly real-time
    - RTF > 1.0 means slower than real-time
    
    Args:
        latency_ms: Processing time in milliseconds
        audio_duration_seconds: Duration of audio in seconds
        
    Returns:
        float: Real-Time Factor
    """
    if audio_duration_seconds <= 0:
        return float('inf')
    
    latency_seconds = latency_ms / 1000.0
    return latency_seconds / audio_duration_seconds


def measure_gpu_memory(
    model,
    sample_input: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure peak GPU memory usage during inference.
    
    Args:
        model: The ASRLLM model
        sample_input: Dict with model inputs
        device: Device (must be "cuda")
        
    Returns:
        Dict with memory statistics:
            - peak_memory_mb: Peak memory allocated in MB
            - peak_memory_gb: Peak memory allocated in GB
            - current_memory_mb: Current memory after inference
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "peak_memory_mb": 0.0,
            "peak_memory_gb": 0.0,
            "current_memory_mb": 0.0,
            "device": "cpu (no GPU memory tracking)",
        }
    
    model.eval()
    
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in sample_input.items()}
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Get baseline memory
    baseline_memory = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        # Run inference
        _ = model(**inputs)
        
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    current_memory = torch.cuda.memory_allocated()
    
    # Convert to MB and GB
    peak_mb = peak_memory / (1024 ** 2)
    peak_gb = peak_memory / (1024 ** 3)
    current_mb = current_memory / (1024 ** 2)
    baseline_mb = baseline_memory / (1024 ** 2)
    
    return {
        "peak_memory_mb": float(peak_mb),
        "peak_memory_gb": float(peak_gb),
        "current_memory_mb": float(current_mb),
        "baseline_memory_mb": float(baseline_mb),
        "inference_memory_mb": float(peak_mb - baseline_mb),  # Memory used by inference
    }


def get_audio_duration_from_mel(
    mel_spectrogram: torch.Tensor,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> float:
    """
    Calculate audio duration from mel spectrogram dimensions.
    
    Whisper uses:
    - sample_rate = 16000 Hz
    - hop_length = 160 samples (10ms per frame)
    - n_fft = 400
    
    Args:
        mel_spectrogram: Tensor of shape (batch, n_mels, n_frames) or (n_mels, n_frames)
        hop_length: Hop length used in STFT (default: 160 for Whisper)
        sample_rate: Audio sample rate (default: 16000 for Whisper)
        
    Returns:
        float: Audio duration in seconds
    """
    # Get number of frames (time dimension)
    if mel_spectrogram.dim() == 3:
        n_frames = mel_spectrogram.shape[2]  # (batch, n_mels, n_frames)
    elif mel_spectrogram.dim() == 2:
        n_frames = mel_spectrogram.shape[1]  # (n_mels, n_frames)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {mel_spectrogram.dim()}D")
    
    # Calculate duration: n_frames * hop_length / sample_rate
    duration_seconds = (n_frames * hop_length) / sample_rate
    
    return duration_seconds


def run_efficiency_benchmark(
    model,
    encoder,
    sample_input: Dict[str, torch.Tensor],
    num_layers: int = None,
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Run complete efficiency benchmark for the model.
    
    This is the main function to call for collecting all efficiency metrics
    for your paper's comparison table.
    
    Args:
        model: The ASRLLM model
        encoder: The Whisper encoder (for parameter counting)
        sample_input: Dict with model inputs including 'audio_mel'
        num_layers: Number of encoder layers being used
        device: Device to run on
        num_warmup: Warmup runs for latency measurement
        num_runs: Number of runs for latency averaging
        
    Returns:
        Dict with all efficiency metrics:
            - params: Parameter count dict
            - latency: Latency statistics dict
            - memory: Memory usage dict
            - rtf: Real-Time Factor
            - audio_duration_s: Duration of test audio
    """
    # 1. Count parameters
    params = count_encoder_parameters(encoder, num_layers)
    
    # 2. Get audio duration from mel spectrogram
    audio_mel = sample_input.get('audio_mel')
    if audio_mel is not None:
        audio_duration = get_audio_duration_from_mel(audio_mel)
    else:
        audio_duration = 1.0  # Default fallback
    
    # 3. Measure latency
    latency = measure_inference_latency(
        model, sample_input, device, num_warmup, num_runs
    )
    
    # 4. Compute RTF
    rtf = compute_rtf(latency["mean_latency_ms"], audio_duration)
    
    # 5. Measure GPU memory
    memory = measure_gpu_memory(model, sample_input, device)
    
    return {
        "params": params,
        "latency": latency,
        "memory": memory,
        "rtf": rtf,
        "audio_duration_s": audio_duration,
        "summary": {
            "encoder_params_M": params["used_params"] / 1e6,
            "latency_ms": latency["mean_latency_ms"],
            "rtf": rtf,
            "memory_gb": memory["peak_memory_gb"],
        }
    }