import cv2
import numpy as np
import imageio
import os

def extract_keyframes(video_path, output_path='summary_video.mp4', frame_rate=5):
    """
    Extracts keyframes from a video based on frame-to-frame difference in HSV space.
    Uses a weighted moving average filter and a statistical threshold to select keyframes.
    
    """

    # Open video and initialize reader
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diff_vector = []

    # Read the first frame and convert to HSV
    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

    # Compute frame-to-frame difference vector
    for _ in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = hsv_frame.astype(float) - prev_hsv.astype(float)
        diff_norm = np.sqrt(np.sum(diff**2))
        diff_vector.append(diff_norm)
        prev_hsv = hsv_frame

    cap.release()

    # Apply weighted moving average filter
    window_len = 11
    weights = [0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
    half = window_len // 2
    smoothed = []

    for k in range(len(diff_vector)):
        center_value = diff_vector[k] * weights[half]

        left_indices = range(max(0, k - half), k)
        right_indices = range(k + 1, min(len(diff_vector), k + half + 1))

        part1 = [diff_vector[i] * weights[half - (k - i)] for i in left_indices]
        part2 = [diff_vector[i] * weights[half + (i - k)] for i in right_indices]

        total_weight = sum(weights[half - (k - i)] for i in left_indices) + \
                       weights[half] + \
                       sum(weights[half + (i - k)] for i in right_indices)

        mean_weighted = (sum(part1) + center_value + sum(part2)) / total_weight if total_weight > 0 else 0
        smoothed.append(mean_weighted)

    # Threshold to detect keyframes
    smoothed = np.array(smoothed)
    threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
    key_indices = np.where(smoothed >= threshold)[0]

    # Write selected keyframes to output video
    cap = cv2.VideoCapture(video_path)
    writer = imageio.get_writer(output_path, fps=frame_rate)

    for idx in key_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

    cap.release()
    writer.close()

    print("Video creation complete.")
    return key_indices.tolist()
