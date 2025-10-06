import numpy as np
import imageio.v3 as iio
from typing import Tuple, List
from PIL import Image

def get_video_info(path: str) -> Tuple[float, float, Tuple[int, int]]:
    """Return (duration_sec, fps, (width, height))."""
    try:
        # Try to get metadata
        meta = iio.immeta(path, plugin="pyav")
        fps = meta.get('fps', 30.0)
    except:
        # Fallback to default fps
        fps = 30.0
    
    # Read first frame to get dimensions
    frame = iio.imread(path, plugin="pyav", index=0)
    height, width = frame.shape[:2]
    
    # Get total frame count
    props = iio.improps(path, plugin="pyav")
    total_frames = props.shape[0]
    
    # Calculate duration
    duration_sec = total_frames / fps if fps > 0 else 0
    
    return duration_sec, fps, (width, height)


def sample_frames(video_path: str, fps: float = None, num_frames: int = None, 
                  resize: int = None, duration: float = None) -> np.ndarray:
    """
    Sample frames uniformly from a video.
    
    Args:
        video_path: Path to video file
        fps: Frames per second (used with duration to calculate num_frames)
        num_frames: Number of frames to sample (overrides fps*duration calculation)
        resize: Target size to resize frames (square: resize x resize)
        duration: Duration in seconds (used with fps to calculate num_frames)
        
    Returns:
        np.ndarray of shape (num_frames, height, width, 3) with RGB frames
    """
    # Get total frame count
    props = iio.improps(video_path, plugin="pyav")
    total_frames = props.shape[0]
    
    # Calculate number of frames to sample
    if num_frames is None:
        if fps is not None and duration is not None:
            num_frames = int(fps * duration)
        else:
            num_frames = 16  # Default
    
    # Calculate frame indices to sample
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    # Read frames one by one (more memory efficient)
    for idx in frame_indices:
        frame = iio.imread(video_path, plugin="pyav", index=int(idx))
        
        # Resize if requested
        if resize is not None:
            img = Image.fromarray(frame)
            img = img.resize((resize, resize), Image.BILINEAR)
            frame = np.array(img)
        
        frames.append(frame)
    
    # If we got fewer frames than requested, pad with the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    
    return np.array(frames)