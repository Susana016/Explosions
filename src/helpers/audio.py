"""
Audio extraction and processing utilities - Fixed for NumPy 2.x compatibility
"""

import numpy as np
import librosa
import tempfile
import subprocess
from pathlib import Path

def extract_audio_array(video_path: str, sr: int = 16000) -> np.ndarray:
    """Extract audio from video file using librosa"""
    try:
        audio, _ = librosa.load(video_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"Warning: Could not extract audio from {video_path}, returning silence")
        return np.zeros(sr, dtype=np.float32)

def log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 64,
    win_length: int = None
) -> np.ndarray:
    """Compute log mel-spectrogram from audio waveform"""
    if len(audio) == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    
    # Convert to integers (in case they come as floats from argparse)
    sr = int(sr)
    n_fft = int(n_fft)
    hop_length = int(hop_length)
    n_mels = int(n_mels)
    
    if win_length is None:
        win_length = n_fft
    else:
        win_length = int(win_length)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, power=2.0
    )
    
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.astype(np.float32)
