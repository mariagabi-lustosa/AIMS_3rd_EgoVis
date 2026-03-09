from pathlib import Path
import numpy as np
import cv2

def read_frame(video_path, frame_index):
    """
    Video frame reader.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()

    return frame_bgr[:, :, ::-1] # frame RGB