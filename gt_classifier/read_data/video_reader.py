from pathlib import Path
import numpy as np
import cv2

def read_frame(video_path, frame_index):
    """
    Video frame reader.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        raise ValueError(f"Failed to read frame {frame_index} from {video_path} (total frames: {total_frames})")
    
    return frame_bgr[:, :, ::-1] # frame RGB