import cv2
import imageio

def create_incomplete_video(video_path, output_path='incomplete_video.mp4', frame_rate=20):
    """
    Creates an incomplete version of the input video by dropping every other frame.
    This is useful for testing interpolation or frame-reconstruction algorithms.
       
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    writer = imageio.get_writer(output_path, fps=frame_rate)
    frame_idx = 0
    written_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 2 == 1:  # Keep only even-numbered frames (1-based indexing)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            written_frames += 1

        frame_idx += 1

    cap.release()
    writer.close()

    print(f"Incomplete video created with {written_frames} frames.")
    return written_frames
