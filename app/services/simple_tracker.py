import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class SimpleTracker:
    """
    A simple object tracker using IoU (Intersection over Union) matching.
    This is a lightweight alternative to ByteTrack.
    """

    def __init__(self, max_lost=50, iou_threshold=0.3):
        self.max_lost = max_lost  # Maximum frames to keep lost track
        self.iou_threshold = iou_threshold  # Minimum IoU for matching
        self.tracks = {}  # Active tracks
        self.track_id_count = 0  # Counter for unique track IDs
        self.lost_tracks = {}  # Tracks that are temporarily lost
        self.frame_count = 0

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        Boxes are in format [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0

        return intersection / union

    def calculate_iou_matrix(self, detections, tracks):
        """
        Calculate IoU matrix between detections and existing tracks.
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.zeros((len(detections), len(tracks)))

        iou_matrix = np.zeros((len(detections), len(tracks)))

        for i, det in enumerate(detections):
            for j, track_id in enumerate(tracks.keys()):
                track = tracks[track_id]
                iou_matrix[i, j] = self.calculate_iou(det[:4], track['bbox'])

        return iou_matrix

    def update(self, detections):
        """
        Update tracker with new detections.

        detections: List of [x1, y1, x2, y2, confidence]
        Returns: List of [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        if len(detections) == 0:
            detections = []

        # Convert to numpy array if needed
        if isinstance(detections, list) and len(detections) > 0:
            detections = np.array(detections)
        elif isinstance(detections, np.ndarray) and len(detections) == 0:
            detections = np.array([])

        tracked_objects = []

        if len(detections) == 0:
            # No detections, update lost count for all tracks
            tracks_to_remove = []
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_count'] += 1
                if self.tracks[track_id]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.tracks[track_id]

            return tracked_objects

        # If we have existing tracks, match them with detections
        if len(self.tracks) > 0:
            # Calculate IoU matrix
            iou_matrix = self.calculate_iou_matrix(detections, self.tracks)

            # Use Hungarian algorithm for optimal matching
            if iou_matrix.size > 0:
                # Convert IoU to cost (1 - IoU)
                cost_matrix = 1 - iou_matrix

                # Solve assignment problem
                det_indices, track_indices = linear_sum_assignment(cost_matrix)

                matched_detections = set()
                matched_tracks = set()
                track_keys = list(self.tracks.keys())

                # Process matches
                for det_idx, track_idx in zip(det_indices, track_indices):
                    if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                        track_id = track_keys[track_idx]
                        # Update existing track
                        self.tracks[track_id]['bbox'] = detections[det_idx][:4]
                        self.tracks[track_id]['confidence'] = detections[det_idx][4]
                        self.tracks[track_id]['lost_count'] = 0

                        matched_detections.add(det_idx)
                        matched_tracks.add(track_id)

                        # Add to output
                        bbox = self.tracks[track_id]['bbox']
                        tracked_objects.append([
                            int(bbox[0]), int(bbox[1]),
                            int(bbox[2]), int(bbox[3]),
                            track_id
                        ])

                # Handle unmatched tracks (lost tracks)
                tracks_to_remove = []
                for track_id in list(self.tracks.keys()):  # Create a copy of keys
                    if track_id not in matched_tracks:
                        self.tracks[track_id]['lost_count'] += 1
                        if self.tracks[track_id]['lost_count'] <= self.max_lost:
                            # Still keep the track but mark as lost
                            bbox = self.tracks[track_id]['bbox']
                            tracked_objects.append([
                                int(bbox[0]), int(bbox[1]),
                                int(bbox[2]), int(bbox[3]),
                                track_id
                            ])
                        else:
                            # Mark track for removal
                            tracks_to_remove.append(track_id)

                # Remove tracks after iteration
                for track_id in tracks_to_remove:
                    del self.tracks[track_id]

                # Handle unmatched detections (new tracks)
                for det_idx in range(len(detections)):
                    if det_idx not in matched_detections:
                        # Create new track
                        self.tracks[self.track_id_count] = {
                            'bbox': detections[det_idx][:4],
                            'confidence': detections[det_idx][4],
                            'lost_count': 0,
                            'age': 0
                        }

                        bbox = detections[det_idx][:4]
                        tracked_objects.append([
                            int(bbox[0]), int(bbox[1]),
                            int(bbox[2]), int(bbox[3]),
                            self.track_id_count
                        ])

                        self.track_id_count += 1
            else:
                # No existing tracks, all detections are new
                for det in detections:
                    self.tracks[self.track_id_count] = {
                        'bbox': det[:4],
                        'confidence': det[4],
                        'lost_count': 0,
                        'age': 0
                    }

                    bbox = det[:4]
                    tracked_objects.append([
                        int(bbox[0]), int(bbox[1]),
                        int(bbox[2]), int(bbox[3]),
                        self.track_id_count
                    ])

                    self.track_id_count += 1
        else:
            # No existing tracks, create new ones for all detections
            for det in detections:
                self.tracks[self.track_id_count] = {
                    'bbox': det[:4],
                    'confidence': det[4],
                    'lost_count': 0,
                    'age': 0
                }

                bbox = det[:4]
                tracked_objects.append([
                    int(bbox[0]), int(bbox[1]),
                    int(bbox[2]), int(bbox[3]),
                    self.track_id_count
                ])

                self.track_id_count += 1

        return tracked_objects

    def reset(self):
        """Reset the tracker state."""
        self.tracks = {}
        self.track_id_count = 0
        self.lost_tracks = {}
        self.frame_count = 0