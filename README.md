# DeepVision Frame Reconstruction System

This repository contains the source code for my Bachelor's final project titled **DeepVision Frame Reconstruction System**, which focuses on intelligent video frame reconstruction using classical computer vision and optical flow techniques.

## Project Overview

The project is structured into three major components:

### 1. Keyframe Extraction
Detects the most significant frames in a video based on HSV color differences. A weighted moving average filter is used to smooth the difference signal, and a statistical threshold identifies keyframes.

### 2. Incomplete Video Generation
Creates a simulated incomplete video by removing every other frame. This step emulates real-world scenarios like transmission loss or low-frame-rate capture.

### 3. Frame Reconstruction via Interpolation
Missing frames are reconstructed by estimating motion between remaining frames using the Farneback optical flow algorithm. Intermediate frames are generated and inserted to produce a complete, smooth video.

## Current Methodology

This system currently uses classical computer vision and image processing methods, including:

- HSV color analysis for frame comparison
- Optical flow for motion estimation
- Interpolation-based reconstruction for missing frames

No deep learning models are used in this version, but the architecture is designed to support future integration with neural-based approaches.

## Applications

This project serves as a foundation for various real-world and research applications:

- Recovery of damaged or incomplete video streams
- Frame rate upscaling (e.g., 30fps to 60fps)
- Video summarization using keyframes
- Preprocessing for deep learning models (temporal consistency, frame interpolation)
- Restoration in surveillance, drone, and medical video footage

### Key Use Case: AI-Oriented Video Dataset Preparation

DeepVision helps in generating clean, temporally consistent video sequences ideal for training AI models in:

- Action recognition
- Gesture detection
- Temporal segmentation
- Video captioning and understanding

## Folder Structure

```
project/
│
├── main.py                          # Entry point for executing the full pipeline
├── keyframe_extraction.py          # Detects keyframes using HSV differences
├── make_incomplete_video.py        # Drops every other frame
├── optical_flow_interpolation.py   # Interpolates missing frames
├── reconstruct_full_video.py       # Builds final output video
├── README.md                       # This documentation file
└── output/                         # Folder to store results
```

## Future Plans

The next stages of development include:

- Replacing classical methods with AI-powered frame generation (e.g., RIFE, Super SloMo)
- Real-time frame interpolation and restoration
- Evaluation on real-world, low-quality video datasets

## How to Run

To run the entire pipeline:

```bash
python main.py
```

Make sure to update the input video path in `main.py` before running.

## Requirements

- Python 3.7 or higher
- OpenCV
- NumPy
- SciPy
- ImageIO

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Author

Shady Nikooei  
Final Year B.Sc. Student in Computer Engineering


This repository will continue to evolve with future integration of deep learning techniques for advanced video restoration and understanding.
