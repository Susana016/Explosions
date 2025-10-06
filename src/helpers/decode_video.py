import numpy as np
import imageio.v3 as iio
from typing import Tuple, List

def get_video_info(path: str) -> Tuple[float, float, Tuple[int, int]]:
    """Return (duration_sec, fps, (width, height))."""
    # Get video properties
    props = iio.improps(path, plugin="pyav")
    
    fps = props.fps
    duration_sec = props.duration
    shape = props.shape  # (frames, height, width, channels)
    
    height, width = shape[1], shape[2]
    
    return duration_sec, fps, (width, height)


def sample_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Sample frames uniformly from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        
    Returns:
        np.ndarray of shape (num_frames, height, width, 3) with RGB frames
    """
    # Read video metadata
    props = iio.improps(video_path, plugin="pyav")
    total_frames = props.shape[0]  # First dimension is number of frames
    
    # Calculate frame indices to sample
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    # Read the entire video
    video = iio.imread(video_path, plugin="pyav")
    
    # Sample the specified frames
    for idx in frame_indices:
        frames.append(video[idx])
    
    # If we got fewer frames than requested, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.array(frames)