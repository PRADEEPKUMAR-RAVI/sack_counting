import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque


class LineCrossingCounter:
    """
    An improved service to count objects crossing a horizontal line in a video.
    Tracks objects moving across the line and counts them as 'in' (top to bottom)
    or 'out' (bottom to top) with enhanced accuracy features.
    """

    def __init__(self, line_position: float = 0.5, frame_height: Optional[int] = None,
                 use_bottom_point: bool = False, history_size: int = 5,
                 min_frames_before_cross: int = 2, smoothing: bool = True):
        """
        Initialize the line crossing counter with improved tracking.

        Args:
            line_position: Position of the line as a fraction of frame height (0.0 to 1.0)
                          0.5 means middle of the frame
            frame_height: Optional frame height to set the line position in pixels
            use_bottom_point: If True, use bottom center of bbox instead of centroid
            history_size: Number of previous positions to track for each object
            min_frames_before_cross: Minimum frames an object must be tracked before counting
            smoothing: Whether to apply position smoothing for stability
        """
        self.line_position = line_position  # Relative position (0-1)
        self.line_y = None  # Absolute pixel position
        if frame_height:
            self.line_y = int(frame_height * line_position)

        # Configuration
        self.use_bottom_point = use_bottom_point
        self.history_size = history_size
        self.min_frames_before_cross = min_frames_before_cross
        self.smoothing = smoothing

        # Counters
        self.in_count = 0  # Objects moving from top to bottom
        self.out_count = 0  # Objects moving from bottom to top

        # Enhanced tracking with history
        self.track_history = {}  # {track_id: {'positions': deque, 'frame_count': int, 'last_side': str}}
        self.crossed_ids = {}  # {track_id: {'direction': str, 'frame': int}} - Never removed until reset
        self.frame_counter = 0  # Global frame counter

    def set_line_position(self, frame_height: int, position: Optional[float] = None):
        """
        Set the absolute line position based on frame height.

        Args:
            frame_height: Height of the video frame
            position: Optional new relative position (0-1)
        """
        if position is not None:
            self.line_position = position
        self.line_y = int(frame_height * self.line_position)

    def get_tracking_point(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Get the point to track for line crossing (centroid or bottom center).

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            (x, y): Tracking point coordinates
        """
        x1, y1, x2, y2 = bbox

        if self.use_bottom_point:
            # Use bottom center of bbox
            x = int((x1 + x2) / 2)
            y = y2  # Bottom of the box
        else:
            # Use centroid
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)

        return x, y

    def smooth_position(self, positions: Deque[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Apply smoothing to position history to reduce jitter.

        Args:
            positions: Deque of (x, y) positions

        Returns:
            Smoothed (x, y) position
        """
        if len(positions) == 0:
            return None

        if not self.smoothing or len(positions) == 1:
            return positions[-1]

        # Use weighted average with more weight on recent positions
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights = weights / weights.sum()

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        smooth_x = int(np.average(xs, weights=weights))
        smooth_y = int(np.average(ys, weights=weights))

        return smooth_x, smooth_y

    def interpolate_missing_frames(self, track_id: int, current_pos: Tuple[int, int],
                                  max_gap: int = 5) -> List[Tuple[int, int]]:
        """
        Interpolate positions for missing frames to handle low FPS or detection gaps.

        Args:
            track_id: The track ID
            current_pos: Current position
            max_gap: Maximum frames to interpolate

        Returns:
            List of interpolated positions
        """
        if track_id not in self.track_history or len(self.track_history[track_id]['positions']) == 0:
            return [current_pos]

        last_pos = self.track_history[track_id]['positions'][-1]
        frame_gap = self.frame_counter - self.track_history[track_id]['last_frame']

        if frame_gap <= 1:
            return [current_pos]

        if frame_gap > max_gap:
            # Too large gap, don't interpolate
            return [current_pos]

        # Linear interpolation
        interpolated = []
        for i in range(1, min(frame_gap, max_gap)):
            alpha = i / frame_gap
            x = int(last_pos[0] + alpha * (current_pos[0] - last_pos[0]))
            y = int(last_pos[1] + alpha * (current_pos[1] - last_pos[1]))
            interpolated.append((x, y))

        interpolated.append(current_pos)
        return interpolated

    def check_crossing(self, track_id: int, current_y: int, previous_y: int) -> Optional[str]:
        """
        Check if an object crossed the line and in which direction.

        Args:
            track_id: The track ID
            current_y: Current Y position
            previous_y: Previous Y position

        Returns:
            'in' if crossed top to bottom, 'out' if bottom to top, None if no crossing
        """
        if self.line_y is None:
            return None

        # Already counted this track
        if track_id in self.crossed_ids:
            return None

        # Need minimum tracking history before counting
        if track_id not in self.track_history:
            return None

        if self.track_history[track_id]['frame_count'] < self.min_frames_before_cross:
            return None

        # Check crossing with some tolerance
        tolerance = 2  # pixels

        # Crossing from top to bottom (IN)
        if previous_y < self.line_y - tolerance and current_y >= self.line_y + tolerance:
            return 'in'

        # Crossing from bottom to top (OUT)
        if previous_y > self.line_y + tolerance and current_y <= self.line_y - tolerance:
            return 'out'

        return None

    def update(self, tracks: List[List[int]], frame_shape: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Update the counter with new tracked objects using improved tracking.

        Args:
            tracks: List of tracked objects, each as [x1, y1, x2, y2, track_id]
            frame_shape: Optional (height, width) to set line position on first frame

        Returns:
            Dictionary containing counting information and statistics
        """
        self.frame_counter += 1

        # Set line position if not already set
        if self.line_y is None and frame_shape is not None:
            self.set_line_position(frame_shape[0])

        current_positions = {}
        current_track_ids = set()
        crossing_events = []

        for track in tracks:
            if len(track) < 5:
                continue

            x1, y1, x2, y2, track_id = track[:5]
            track_id = int(track_id)
            current_track_ids.add(track_id)

            # Get tracking point
            x, y = self.get_tracking_point([x1, y1, x2, y2])
            current_positions[track_id] = (x, y)

            # Initialize track if new
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'positions': deque(maxlen=self.history_size),
                    'frame_count': 0,
                    'last_frame': self.frame_counter,
                    'last_side': 'above' if y < self.line_y else 'below'
                }

            # Handle missing frames with interpolation
            interpolated_positions = self.interpolate_missing_frames(track_id, (x, y))

            # Process each position (interpolated or actual)
            for pos in interpolated_positions:
                # Add to history
                self.track_history[track_id]['positions'].append(pos)
                self.track_history[track_id]['frame_count'] += 1

                # Get smoothed position for crossing detection
                if self.smoothing and len(self.track_history[track_id]['positions']) > 1:
                    smooth_pos = self.smooth_position(self.track_history[track_id]['positions'])
                    check_y = smooth_pos[1]
                else:
                    check_y = pos[1]

                # Check for line crossing
                if len(self.track_history[track_id]['positions']) > 1:
                    prev_y = self.track_history[track_id]['positions'][-2][1]
                    crossing_direction = self.check_crossing(track_id, check_y, prev_y)

                    if crossing_direction:
                        if crossing_direction == 'in':
                            self.in_count += 1
                        else:  # 'out'
                            self.out_count += 1

                        self.crossed_ids[track_id] = {
                            'direction': crossing_direction,
                            'frame': self.frame_counter
                        }

                        crossing_events.append({
                            'track_id': track_id,
                            'direction': crossing_direction,
                            'position': (x, y)
                        })

                        # Update last side
                        self.track_history[track_id]['last_side'] = 'below' if crossing_direction == 'in' else 'above'

            # Update last frame
            self.track_history[track_id]['last_frame'] = self.frame_counter

        # Clean up stale tracks (but keep crossed_ids)
        stale_threshold = 30  # frames
        tracks_to_remove = []

        for track_id in self.track_history:
            if track_id not in current_track_ids:
                frames_since_seen = self.frame_counter - self.track_history[track_id]['last_frame']
                if frames_since_seen > stale_threshold:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            # Note: We intentionally keep crossed_ids to prevent re-counting

        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'line_y': self.line_y,
            'current_positions': current_positions,
            'crossing_events': crossing_events,
            'active_tracks': len(current_track_ids),
            'total_crossed': len(self.crossed_ids)
        }

    def draw_line_and_counts(self, frame, color: Tuple[int, int, int] = (0, 255, 255),
                            thickness: int = 2, show_counts: bool = True,
                            show_direction_arrows: bool = True):
        """
        Draw the counting line, counts, and direction indicators on the frame.

        Args:
            frame: The video frame to draw on
            color: BGR color for the line
            thickness: Line thickness
            show_counts: Whether to display count text
            show_direction_arrows: Whether to show direction arrows

        Returns:
            The frame with drawings (modified in-place)
        """
        if self.line_y is None:
            return frame

        height, width = frame.shape[:2]

        # Draw the horizontal line
        cv2.line(frame, (0, self.line_y), (width, self.line_y), color, thickness)

        # Draw direction arrows if requested
        if show_direction_arrows:
            arrow_x = width - 100
            # Down arrow (IN)
            cv2.arrowedLine(frame, (arrow_x, self.line_y - 30),
                          (arrow_x, self.line_y - 10), (0, 255, 0), 2)
            cv2.putText(frame, "IN", (arrow_x + 10, self.line_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Up arrow (OUT)
            cv2.arrowedLine(frame, (arrow_x, self.line_y + 30),
                          (arrow_x, self.line_y + 10), (0, 0, 255), 2)
            cv2.putText(frame, "OUT", (arrow_x + 10, self.line_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw count information
        if show_counts:
            # Background for better visibility
            cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 90), color, 2)

            # Draw text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"IN: {self.in_count}", (20, 40),
                       font, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (20, 70),
                       font, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"TOTAL: {self.in_count + self.out_count}", (180, 55),
                       font, 0.8, (255, 255, 255), 2)

        return frame

    def draw_tracking_points(self, frame, positions: Optional[Dict[int, Tuple[int, int]]] = None,
                            show_trails: bool = True):
        """
        Draw tracking points and optional trails on the frame.

        Args:
            frame: The video frame to draw on
            positions: Current positions or None to use history
            show_trails: Whether to show position history trails

        Returns:
            The frame with tracking visualization
        """
        # Draw trails if requested
        if show_trails:
            for track_id, info in self.track_history.items():
                if len(info['positions']) > 1:
                    # Draw trail
                    points = np.array(list(info['positions']), dtype=np.int32)
                    for i in range(1, len(points)):
                        # Fade older points
                        alpha = i / len(points)
                        color = (255 * alpha, 0, 255 * (1 - alpha))
                        cv2.line(frame, tuple(points[i-1]), tuple(points[i]),
                               color, 1)

        # Draw current positions
        if positions:
            for track_id, (x, y) in positions.items():
                # Different color based on crossing status
                if track_id in self.crossed_ids:
                    if self.crossed_ids[track_id]['direction'] == 'in':
                        point_color = (0, 255, 0)  # Green for IN
                    else:
                        point_color = (0, 0, 255)  # Red for OUT
                else:
                    point_color = (255, 0, 255)  # Magenta for not crossed

                # Draw point
                cv2.circle(frame, (x, y), 5, point_color, -1)

                # Draw track ID
                cv2.putText(frame, str(track_id), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)

        return frame

    def reset(self):
        """Reset all counters and tracking history."""
        self.in_count = 0
        self.out_count = 0
        self.track_history = {}
        self.crossed_ids = {}
        self.frame_counter = 0

    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about the counting.

        Returns:
            Dictionary with detailed counting statistics
        """
        return {
            'in_count': self.in_count,
            'out_count': self.out_count,
            'total_count': self.in_count + self.out_count,
            'line_position': self.line_position,
            'line_y': self.line_y,
            'active_tracks': len(self.track_history),
            'total_crossed': len(self.crossed_ids),
            'frame_counter': self.frame_counter,
            'crossed_tracks': {
                tid: info for tid, info in self.crossed_ids.items()
            }
        }