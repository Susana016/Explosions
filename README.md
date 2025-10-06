# Science Docu-Drama: Explosion or Explanation — Starter Kit (PyTorch + Streamlit)

A classroom-friendly, **video-as-stream** project. As an MP4 plays, a model ingests short windows of **video frames** and **audio** and predicts whether the **next second** contains **Explosion/Action** (1) or **Explanation/Calm** (0). The app runs inference offline and displays a probability timeline.

**Estimated Duration:** 3 hours (setup: 30 min, training: 1 hour, experimentation: 1.5 hours)

## Learning Objectives
By completing this project, students will:
- Understand multimodal machine learning (video + audio fusion)
- Build and train a CNN-based binary classifier with PyTorch
- Work with video/audio preprocessing pipelines
- Create an interactive ML demo using Streamlit
- Practice data annotation and dataset construction
- Experiment with model ablations and hyperparameter tuning

## Why this works
- Two data types (video frames + audio mel-spectrogram) fused with a simple CNN → **late fusion**
- **Humorous**: "Explosion or Explanation?" rapidly becomes a running joke during demo
- **Divergent thinking**: window sizes, motion vs. raw frames, audio feature choices, fusion tweaks
- Stack: **PyTorch** for training; **Streamlit** for an interactive inference UI

## Deliverables
At the end of this activity, students should submit:
1. Trained model weights (`weights/best.pt`)
2. Annotation CSV file(s) for their chosen video(s)
3. Screenshot or recording of the Streamlit app showing predictions
4. Brief report (1-2 paragraphs) describing:
   - What video(s) they used and why
   - Training results (loss, AUC, number of windows)
   - At least one ablation experiment they tried and its impact

---

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes depending on your connection speed. PyTorch packages are large.

### 2) Obtain video clips
You need short MP4 video clips (30-120 seconds recommended) containing both action/explosion scenes and calm/explanation scenes.

**Suggested sources:**
- YouTube science documentaries (e.g., MythBusters, Science Channel clips)
- Movie trailers with action sequences
- Educational science videos with experiments
- Personal recordings of demonstrations

**Important:** Keep video files locally. Use `youtube-dl` or `yt-dlp` to download clips:
```bash
# Example: download a 2-minute clip
yt-dlp -f "best[height<=480]" --download-sections "*0:00-2:00" "VIDEO_URL" -o "my_video.mp4"
```

Place MP4 files in a directory of your choice (e.g., `videos/`).

### 3) Create annotation CSV files
For each video, create an annotation CSV in `data/sample_annotations/` that matches the video filename.

**Example:** For `videos/mythbusters_clip.mp4`, create `data/sample_annotations/mythbusters_clip.csv`

**CSV Format:**
```csv
video_id,start_sec,end_sec,event_type
mythbusters_clip,2.0,2.3,explanation
mythbusters_clip,7.5,7.6,explosion
mythbusters_clip,10.0,11.5,explanation
mythbusters_clip,18.2,18.3,explosion
```

**Annotation Guidelines:**
- `video_id` must match the video filename (without `.mp4` extension)
- `start_sec` and `end_sec` define the time range of each event (in seconds)
- `event_type` should be either `explosion` (action/intense moments) or `explanation` (calm/talking segments)
- **Labeling strategy:** Watch your video and mark timestamps where explosions/action occur vs. when people are explaining/demonstrating calmly
- You can use VLC media player's time display to find exact timestamps
- Reference the provided `data/sample_annotations/sample_trailer.csv` as an example

**Binary classification:** The model treats `explosion` as the positive class (1) and everything else as negative (0).

### 4) Precompute training windows (recommended for speed)
```bash
python src/precompute_windows.py \
  --videos "videos/*.mp4" \
  --ann_dir data/sample_annotations \
  --out_file data/dataset_windows.pt \
  --fps 4 --win_sec 2.0 --stride_sec 0.5 \
  --sr 16000 --n_mels 64 --horizon_sec 1.0
```

**What this does:**
- Extracts video frames at low FPS (4 frames/second by default)
- Computes frame differences to capture motion
- Extracts audio and converts to log-mel spectrograms
- Creates sliding windows of frames + audio features
- Labels windows based on whether an explosion occurs in the next `horizon_sec` seconds
- Saves everything to a single `.pt` file for fast training

**Expected output:** You should see a message like:
```
Saved: data/dataset_windows.pt | vids=(N, 3, 224, 224) mels=(N, 1, 64, T) labels=(N,)
```
Where `N` is the number of windows extracted (typically 50-200 per minute of video).

### 5) Train a baseline model
```bash
python src/train.py --dataset_pt data/dataset_windows.pt --epochs 5 --batch_size 32 --lr 2e-4
```

**What this does:**
- Loads the preprocessed windows from the `.pt` file
- Splits data into train/validation sets (80/20 by default)
- Trains a multimodal CNN (ResNet18 for video + small CNN for audio)
- Saves the best model based on validation AUC to `weights/best.pt`

**Expected output:** You should see training progress like:
```
epoch 1 | train_loss 0.6543 | val_loss 0.6234 | val_auc 0.623
✓ Saved new best to weights/best.pt
epoch 2 | train_loss 0.5821 | val_loss 0.5891 | val_auc 0.701
✓ Saved new best to weights/best.pt
...
```

**Training time:**
- CPU: 5-15 minutes (depending on dataset size)
- GPU: 1-3 minutes

**Note:** The `weights/` directory will be created automatically if it doesn't exist.

### 6) Run the Streamlit app (interactive inference)
```bash
streamlit run src/app.py
```

**What this does:**
- Launches an interactive web app (opens in your browser automatically)
- Loads your trained model (`weights/best.pt`)
- Allows you to upload any MP4 video for inference
- Displays a probability timeline showing "explosion likelihood" over time
- Highlights moments where the model predicts high probability of explosions

**Using the app:**
1. Upload an MP4 file using the file uploader
2. Adjust parameters:
   - **Sampling FPS**: Lower = faster processing, higher = more detail
   - **Window (sec)**: Duration of input window (2.0s recommended)
   - **Stride (sec)**: How often to sample windows (0.5s = 50% overlap)
   - **Alert threshold**: Probability threshold for flagging "explosion" regions
3. View the probability timeline chart
4. Experiment with different threshold values to see how it affects predictions

**Expected result:** You should see a line chart with probability values between 0 and 1, with peaks at moments the model thinks contain explosions.

---

## Project Structure

```
explosion_or_explanation_starter_kit/
  README.md
  requirements.txt
  src/
    app.py
    train.py
    precompute_windows.py
    models/explex_net.py
    helpers/decode_video.py
    helpers/audio.py
    helpers/windows.py
    helpers/datasets.py
    helpers/metrics.py
  data/
    dataset_windows.pt        # created by precompute script
    sample_annotations/
      sample_trailer.csv      # example annotations format (no video provided)
  weights/
    (best.pt will be saved here after training)
```

---

## Experimental Ideas (Choose at least 1)

After getting the baseline working, students should experiment with modifications to improve or understand the model better:

### Architecture Ablations
1. **Frame representation:**
   - Current: Frame differences (motion cues)
   - Try: Raw RGB frames
   - Edit: `src/precompute_windows.py` and `src/app.py` (change `diff_stacks` to use raw `frames`)

2. **Audio features:**
   - Current: 64 mel bins
   - Try: 40 or 80 mel bins, different hop sizes
   - Edit: `--n_mels` parameter in precompute script

3. **Fusion strategy:**
   - Current: Simple concatenation (late fusion)
   - Try: Gated fusion (learnable weights per modality)
   - Edit: `src/models/explex_net.py` - modify the `forward()` method to add scalar weights

4. **Vision backbone:**
   - Current: Frozen ResNet18
   - Try: Fine-tuning the backbone
   - Edit: `src/models/explex_net.py` - set `train_backbone=True` in `VisionCNN`

### Hyperparameter Tuning
- Window size (`--win_sec`): Try 1.0, 2.0, 3.0, 4.0
- Stride (`--stride_sec`): Try 0.2, 0.5, 1.0 (smaller = more overlap)
- Learning rate (`--lr`): Try 1e-4, 2e-4, 5e-4
- Prediction horizon (`--horizon_sec`): Try 0.5, 1.0, 2.0 seconds

### Analysis Questions
- Which modality (video or audio) is more important? Can you disable one?
- How does the model perform on videos it wasn't trained on?
- What happens if you train on action movies and test on documentaries?
- Can you visualize which frames or audio segments the model focuses on?

## Teaching Notes (For Instructors)

- **Positive window definition**: a window is positive if an `explosion` event begins **within the next `horizon_sec`** after the window end. (Default 1.0 s.)
- **Data leakage prevention**: When splitting train/val, the code currently uses random split. For production, split **by video id**, not by window indices.
- **Fast baseline design**: frozen vision backbone (ResNet18); small 2D Audio-CNN; simple concatenation fusion
- **Time management**:
  - Setup + annotation: 30 min
  - Training baseline: 10-20 min
  - Experimentation: 1-2 hours
  - Demo + writeup: 30 min
- **Ethics reminder**: Discourage uploading copyrighted long-form content to public repos. Use short clips (under 2 minutes) or Creative Commons videos.
- **Performance note**: Streamlit app computes the full probability timeline before display (no real-time streaming). This is intentional for simplicity.

---

## Troubleshooting

### Common Issues

**1. "No videos matched the glob pattern"**
- Check your path in `--videos` parameter
- Use quotes around the glob pattern: `"videos/*.mp4"`
- Verify MP4 files exist in the specified directory

**2. "Annotation CSV not found for {video_id}"**
- The CSV filename must match the video filename (without `.mp4`)
- Example: `my_video.mp4` needs `data/sample_annotations/my_video.csv`
- Check that `video_id` column in CSV matches the filename

**3. "FileNotFoundError: weights/best.pt"**
- You need to train the model first using `python src/train.py ...`
- The Streamlit app expects trained weights to exist
- Alternatively, the app will warn you and use random weights (won't produce good predictions)

**4. MP4 audio fails to load**
- We use **MoviePy** as the default backend (bundles FFmpeg via `imageio-ffmpeg`)
- Ensure all packages in `requirements.txt` are installed
- Try re-encoding your video: `ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4`

**5. Training is very slow**
- Precompute windows first (step 4) instead of loading videos during training
- Reduce `--batch_size` if running out of memory
- Use a GPU if available (automatic in PyTorch)
- Reduce dataset size (use fewer videos or shorter clips)

**6. "No windows were generated"**
- Your video might be too short for the window size
- Try reducing `--win_sec` or increasing clip length
- Check that FPS parameter doesn't exceed video's actual frame rate

**7. Model AUC is ~0.5 (random performance)**
- Dataset might be too small (try more videos or better annotations)
- Class imbalance: ensure you have both explosion and explanation events
- Labels might be noisy: review your annotations
- Try training for more epochs: `--epochs 10`

**8. Streamlit app shows "Connection error"**
- The app should auto-open in your browser at `http://localhost:8501`
- If not, manually open that URL
- Check firewall settings if on a shared network

**9. On Windows: opencv-python GUI dependency issues**
- The `-headless` wheel is included in `requirements.txt`
- If issues persist, uninstall opencv-python and reinstall: `pip uninstall opencv-python && pip install opencv-python-headless`

**10. Out of memory errors**
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--fps` (try 2 instead of 4)
- Use shorter video clips
- Close other applications

---

## Quick Reference: Complete Workflow

Here's a condensed version of the entire workflow once you understand the project:

```bash
# 1. Setup (once)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get videos and annotate
# - Download clips to videos/ directory
# - Create CSV files in data/sample_annotations/
# - Each video needs a matching CSV file

# 3. Preprocess
python src/precompute_windows.py \
  --videos "videos/*.mp4" \
  --ann_dir data/sample_annotations \
  --out_file data/dataset_windows.pt

# 4. Train
python src/train.py --dataset_pt data/dataset_windows.pt --epochs 5

# 5. Demo
streamlit run src/app.py
```

**Success checklist:**
- ✅ Environment created and packages installed
- ✅ At least 1 video file (30+ seconds) downloaded
- ✅ Matching annotation CSV created with both event types
- ✅ Precompute script runs and creates `.pt` file with N > 50 windows
- ✅ Training shows improving AUC (> 0.6 is decent for starters)
- ✅ Streamlit app loads and shows probability timeline
- ✅ At least 1 experimental modification attempted
- ✅ Brief writeup completed describing results

---

## Additional Resources

**Understanding the code:**
- `src/models/explex_net.py` - Model architecture (ResNet18 + Audio CNN + fusion)
- `src/helpers/decode_video.py` - Video frame extraction
- `src/helpers/audio.py` - Audio processing and mel-spectrogram conversion
- `src/helpers/windows.py` - Sliding window creation and labeling logic
- `src/helpers/datasets.py` - PyTorch Dataset wrapper for training

**Learning more:**
- PyTorch tutorials: https://pytorch.org/tutorials/
- Mel-spectrogram explanation: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
- ResNet paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Late fusion in multimodal learning: Combine features from different modalities at the end

---

Happy hacking! 💥📊
