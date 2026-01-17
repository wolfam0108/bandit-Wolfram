"""
Standalone Bandit inference script for original bandit models.
No asteroid dependency - uses local bsrnn_standalone module.

Supports:
- dnr-3s-mus64-l1snr-plus.ckpt
- model_bandit_plus_dnr_sdr_11.47.ckpt
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torchaudio as ta
from tqdm import tqdm
from typing import Dict, List, Optional

# Add current dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import from local bsrnn_standalone
from bsrnn_standalone.wrapper import MultiMaskMultiSourceBandSplitRNN


def detect_model_config(state_dict: Dict) -> Dict:
    """Detect model configuration from state_dict"""
    first_key = list(state_dict.keys())[0]
    
    # Detect prefixes
    has_model = first_key.startswith("model.")
    has_bsrnn = "bsrnn." in first_key
    
    # Detect stems
    stems = set()
    for key in state_dict.keys():
        if "mask_estim" in key:
            parts = key.split(".")
            idx = parts.index("mask_estim")
            if idx + 1 < len(parts):
                stem = parts[idx + 1]
                if stem not in ["norm_mlp", "fc", "norm", "output", "hidden", "freq_weights"]:
                    stems.add(stem)
    
    # Detect n_sqm_modules from seqband indices
    # The model creates n_sqm * 2 seqband modules (seqband + seqtime)
    # So if we see indices 0-15, n_sqm = 8
    sqm_indices = set()
    for key in state_dict.keys():
        if "tf_model.seqband" in key:
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "seqband" and i + 1 < len(parts):
                    try:
                        sqm_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    
    # Divide by 2 because model creates seqband + seqtime for each sqm module
    n_sqm = (max(sqm_indices) + 1) // 2 if sqm_indices else 8
    
    return {
        "has_model_prefix": has_model,
        "has_bsrnn_prefix": has_bsrnn,
        "stems": sorted(stems),
        "n_sqm_modules": n_sqm,
    }


def clean_state_dict(state_dict: Dict, config: Dict) -> Dict:
    """Remove only model. prefix, keep bsrnn. prefix as model expects it"""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        # Only remove model. prefix, keep bsrnn. prefix!
        if config["has_model_prefix"] and new_key.startswith("model."):
            new_key = new_key.replace("model.", "", 1)
        # Don't remove bsrnn. prefix - the model structure expects it
        cleaned[new_key] = value
    return cleaned


def create_model(config: Dict, fs: int = 44100) -> nn.Module:
    """Create Bandit model matching checkpoint config"""
    model = MultiMaskMultiSourceBandSplitRNN(
        in_channel=1,
        stems=config["stems"],
        band_specs="musical",
        n_bands=64,
        fs=fs,
        require_no_overlap=False,
        require_no_gap=True,
        normalize_channel_independently=False,
        treat_channel_as_feature=True,
        n_sqm_modules=config["n_sqm_modules"],
        emb_dim=128,
        rnn_dim=256,
        bidirectional=True,
        rnn_type="GRU",
        mlp_dim=512,
        hidden_activation="Tanh",
        hidden_activation_kwargs=None,
        complex_mask=True,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window_fn="hann_window",
        wkwargs=None,
        power=None,
        center=True,
        normalized=True,
        pad_mode="constant",
        onesided=True,
        use_freq_weights=True,
    )
    return model


def chunked_inference(
    model,
    audio,  # (channels, samples)
    fs=44100,
    chunk_seconds=30.0,
    overlap_seconds=2.0,
    device="cuda",
    use_half=True,
):
    """Chunked inference with overlap and crossfade"""
    n_channels, n_samples = audio.shape
    chunk_samples = int(chunk_seconds * fs)
    overlap_samples = int(overlap_seconds * fs)
    hop_samples = chunk_samples - overlap_samples
    
    stems = model.stems
    outputs = {stem: torch.zeros(n_channels, n_samples) for stem in stems}
    
    fade_in = torch.linspace(0, 1, overlap_samples)
    fade_out = torch.linspace(1, 0, overlap_samples)
    
    n_chunks = max(1, (n_samples - overlap_samples) // hop_samples + 1)
    
    print(f"Processing {n_chunks} chunks")
    
    for i in tqdm(range(n_chunks), desc="Processing"):
        start = i * hop_samples
        end = min(start + chunk_samples, n_samples)
        
        chunk = audio[:, start:end]
        actual_len = chunk.shape[1]
        
        if actual_len < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))
        
        # Process mono (model expects mono input)
        for ch in range(n_channels):
            chunk_mono = chunk[ch:ch+1, :]
            chunk_gpu = chunk_mono[None, :, :].to(device)
            
            batch = {"audio": {"mixture": chunk_gpu}}
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=use_half):
                    _, result = model(batch)
            
            for stem in stems:
                stem_audio = result["audio"][stem][0, 0, :actual_len].float().cpu()
                
                if i == 0:
                    outputs[stem][ch, start:start+actual_len] = stem_audio
                else:
                    overlap_start = start
                    overlap_end = start + overlap_samples
                    
                    if overlap_end <= n_samples:
                        outputs[stem][ch, overlap_start:overlap_end] *= fade_out
                        outputs[stem][ch, overlap_start:overlap_end] += stem_audio[:overlap_samples] * fade_in
                        
                        if overlap_samples < actual_len:
                            outputs[stem][ch, overlap_end:start+actual_len] = stem_audio[overlap_samples:]
                    else:
                        outputs[stem][ch, start:start+actual_len] = stem_audio
        
        torch.cuda.empty_cache()
    
    return outputs


def run_inference(
    checkpoint_path: str,
    audio_path: str,
    output_dir: str = None,
    fs: int = 44100,
    device: str = "cuda",
    use_half: bool = True,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_path), "separated")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n=== Loading: {os.path.basename(checkpoint_path)} ===")
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    config = detect_model_config(state_dict)
    print(f"Stems: {config['stems']}, SQM: {config['n_sqm_modules']}")
    
    cleaned = clean_state_dict(state_dict, config)
    
    model = create_model(config, fs)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded: {len(cleaned)} keys, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    if missing:
        print(f"Warning: {len(missing)} missing keys!")
    
    model = model.to(device)
    model.eval()
    
    # Load audio
    print(f"\n=== Audio: {os.path.basename(audio_path)} ===")
    audio, audio_fs = ta.load(audio_path)
    duration = audio.shape[1] / audio_fs
    print(f"Duration: {duration:.1f}s, Channels: {audio.shape[0]}")
    
    if audio_fs != fs:
        audio = ta.functional.resample(audio, audio_fs, fs)
    
    # Process
    print(f"\n=== Processing ===")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    outputs = chunked_inference(
        model, audio, fs=fs,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        device=device,
        use_half=use_half,
    )
    
    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nDone: {elapsed:.1f}s ({duration/elapsed:.1f}x realtime), VRAM: {peak_vram:.2f}GB")
    
    # Save
    print(f"\n=== Saving ===")
    for stem, audio_out in outputs.items():
        path = os.path.join(output_dir, f"{stem}_estimate.wav")
        ta.save(path, audio_out, fs)
        print(f"  {path}")
    
    print("\nComplete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--audio", "-a", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--chunk", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=2.0)
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        audio_path=args.audio,
        output_dir=args.output,
        device=args.device,
        use_half=not args.no_half,
        chunk_seconds=args.chunk,
        overlap_seconds=args.overlap,
    )
