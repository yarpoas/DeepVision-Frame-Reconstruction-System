import cv2
import imageio
from optical_flow_interpolation import interpolate_frame_optical_flow

def reconstruct_video_with_interpolation(input_path, output_path='complete_video.mp4', output_fps=None):
    """
    Reads an incomplete video, interpolates missing frames, and reconstructs a complete video.
    
    Parameters:
        input_path (str): Path to the input incomplete video.
        output_path (str): Path to the output complete video.
        output_fps (float or None): Frame rate for the output video. If None, uses 1.5x input fps.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + input_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = output_fps if output_fps else original_fps * 1.5
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load all frames into memory
    print("Reading frames into memory...")
    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    print("All frames loaded into memory.")

    # Prepare writer
    writer = imageio.get_writer(output_path, fps=out_fps)
    print("Reconstructing video with interpolated frames...")

    for i in range(len(frames) - 1):
        writer.append_data(frames[i])
        interpolated = interpolate_frame_optical_flow(frames[i], frames[i + 1])
        writer.append_data(interpolated)

    writer.append_data(frames[-1])
    writer.close()

    print("Video reconstruction complete.")
