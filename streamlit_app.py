"""
Streamlit app for Explosion vs Explanation inference
"""

import streamlit as st
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import your helper functions
import sys
sys.path.insert(0, 'src')
from helpers.decode_video import sample_frames, get_video_info
from helpers.audio import extract_audio_array, log_mel_spectrogram
from helpers.windows import make_windows
from models.explex_net import ExplexNet

st.set_page_config(page_title="Explosion or Explanation?", layout="wide")

@st.cache_resource
def load_model(weights_path='weights/best.pt'):
    """Load trained model"""
    if not Path(weights_path).exists():
        st.warning(f"Model weights not found at {weights_path}. Using untrained model.")
        # default to 9-channel frame-diff stacks (3 diffs × 3 RGB channels)
        model = ExplexNet(vid_in_ch=9)
    else:
        model = ExplexNet(vid_in_ch=9)
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
    return model

def process_video(video_path, model, fps=4, win_sec=2.0, stride_sec=0.5, 
                  sr=16000, n_mels=64, mel_hop=0.010):
    """Process video and return predictions"""
    
    # Extract frames
    with st.spinner("Extracting video frames..."):
        frames = sample_frames(video_path, fps=fps, resize=224)
        
        # Create frame differences
        diff_stacks = []
        for i in range(len(frames)):
            if i < 1:
                diff_stacks.append(frames[i].repeat(3,1,1))
            else:
                diff_stacks.append(torch.cat([frames[i]-frames[i-1]]*3, dim=0))
    
    # Extract audio
    with st.spinner("Extracting audio features..."):
        audio = extract_audio_array(video_path, sr=sr)
        mel = log_mel_spectrogram(audio, sr=sr, n_mels=n_mels, 
                                   hop_length=int(mel_hop * sr),
                                   win_length=int(0.025 * sr))
        mel = torch.from_numpy(mel).unsqueeze(0)
    
    # Create windows
    with st.spinner("Creating windows..."):
        vids, mels, centers = make_windows(diff_stacks, mel, fps=fps,
                                           win_sec=win_sec, stride_sec=stride_sec,
                                           mel_hop_s=mel_hop)
    
    if not vids:
        st.error("No windows generated from video")
        return None, None
    
    # Run inference
    with st.spinner("Running model inference..."):
        predictions = []
        model.eval()
        with torch.no_grad():
            for v, m in zip(vids, mels):
                # Add batch dimension
                # frames from sample_frames are already normalized; do not divide by 255 again
                v_batch = v.unsqueeze(0).float()
                m_batch = m.unsqueeze(0).float()

                # Forward pass: model returns a single logit per sample -> use sigmoid
                logits = model(v_batch, m_batch)
                prob = torch.sigmoid(logits).item()
                predictions.append(prob)
    
    return centers, predictions

def plot_timeline(centers, predictions, threshold=0.5):
    """Plot probability timeline"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot probability line
    ax.plot(centers, predictions, linewidth=2, color='#1f77b4', label='Explosion Probability')
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    
    # Highlight regions above threshold
    above_thresh = np.array(predictions) > threshold
    if any(above_thresh):
        ax.fill_between(centers, 0, 1, where=above_thresh, alpha=0.2, color='red', label='Alert Region')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Explosion Detection Timeline', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def main():
    st.title("💥 Explosion or Explanation?")
    st.markdown("Upload a video to detect explosions vs calm explanations using multimodal ML")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    fps = st.sidebar.slider("Sampling FPS", 1, 10, 4, help="Frames per second to sample from video")
    win_sec = st.sidebar.slider("Window (sec)", 0.5, 5.0, 2.0, 0.5, help="Duration of input window")
    stride_sec = st.sidebar.slider("Stride (sec)", 0.1, 2.0, 0.5, 0.1, help="How often to sample windows")
    threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05, help="Probability threshold for explosions")
    
    # Load model
    model = load_model()
    
    # File upload
    uploaded_file = st.file_uploader("Upload MP4 video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Get video info (returns duration, fps, (width, height))
            duration, v_fps, (width, height) = get_video_info(tmp_path)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("Resolution", f"{width}x{height}")
            with col3:
                st.metric("FPS", f"{v_fps:.1f}")
            
            # Process video
            if st.button("Analyze Video", type="primary"):
                centers, predictions = process_video(
                    tmp_path, model, fps=fps, win_sec=win_sec, 
                    stride_sec=stride_sec
                )
                
                if centers is not None:
                    st.success(f"Processed {len(centers)} windows")
                    
                    # Plot timeline
                    fig = plot_timeline(centers, predictions, threshold)
                    st.pyplot(fig)
                    
                    # Statistics
                    st.subheader("Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Max Probability", f"{max(predictions):.3f}")
                    with col2:
                        st.metric("Mean Probability", f"{np.mean(predictions):.3f}")
                    with col3:
                        alerts = sum(1 for p in predictions if p > threshold)
                        st.metric("Alert Windows", alerts)
                    with col4:
                        pct = (alerts / len(predictions)) * 100
                        st.metric("% Above Threshold", f"{pct:.1f}%")
                    
                    # Show high-probability moments
                    st.subheader("High Probability Moments")
                    high_prob_moments = [(c, p) for c, p in zip(centers, predictions) if p > threshold]
                    
                    if high_prob_moments:
                        for time, prob in sorted(high_prob_moments, key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"⚠️ **{time:.1f}s**: {prob:.1%} probability")
                    else:
                        st.info("No moments above threshold detected")
                    
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload a video** using the file uploader above
        2. **Adjust parameters** in the sidebar:
           - Lower FPS = faster processing
           - Longer windows = more context
           - Lower stride = more overlap (smoother predictions)
        3. **Click "Analyze Video"** to run inference
        4. **View the timeline** showing explosion probability over time
        5. **Adjust threshold** to change sensitivity
        
        The model looks at both video frames (visual motion) and audio (loud sounds) 
        to predict whether each moment contains an explosion or explanation.
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info**")
    if Path('weights/best.pt').exists():
        st.sidebar.success("✓ Trained model loaded")
    else:
        st.sidebar.warning("⚠ Using untrained model")

if __name__ == "__main__":
    main()