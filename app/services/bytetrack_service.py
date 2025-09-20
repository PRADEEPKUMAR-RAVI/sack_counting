from app.services.simple_tracker import SimpleTracker
from typing import List

class ByteTrackService:
    def __init__(self, frame_rate=30):
        # Initialize our custom simple tracker
        # max_lost: number of frames to keep a lost track before removing it
        # iou_threshold: minimum IoU for matching detections to existing tracks
        self.tracker = SimpleTracker(max_lost=50, iou_threshold=0.3)
        self.frame_rate = frame_rate

    def update_tracks(self, detections: List[List[float]], img_shape: tuple = None) -> List[List[float]]:
        """
        Update tracker with new detections.

        detections: List of [x1, y1, x2, y2, confidence] from YOLO
        img_shape: Tuple of (height, width) of the frame (not used in simple tracker)
        returns: List of [x1, y1, x2, y2, track_id] for current frame
        """
        # Our simple tracker handles the conversion internally
        tracks = self.tracker.update(detections)
        return tracks

    def reset(self):
        """Reset the tracker if starting a new video"""
        self.tracker.reset()
