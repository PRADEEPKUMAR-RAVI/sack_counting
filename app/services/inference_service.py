
from fastapi import UploadFile, File
from ultralytics import YOLO
from pathlib import Path
from app.services.bytetrack_service import ByteTrackService
from app.services.line_counter_service import LineCrossingCounter
import cv2


class InferenceService:
    def __init__(self):
        self.uploads_dir = Path("uploads")
        self.outputs_dir = Path("outputs")
        self.model = YOLO("/home/softsuave/Documents/Yolo_Models/Learning_model/app/models/best.pt")
        self.tracker = ByteTrackService(frame_rate=30)
        # Improved line counter with better accuracy settings
        self.line_counter = LineCrossingCounter(
            line_position=0.5,  # Line at middle of frame
            use_bottom_point=False,  # Use centroid for tracking
            history_size=10,  # Keep 10 frames of history
            min_frames_before_cross=3,  # Need 3 frames before counting
            smoothing=True  # Enable position smoothing
        )
    def process_video(self, file: UploadFile = File(...)):
        try:
            if file.content_type not in ["video/mp4"]:
                raise ValueError("Unsupported file type. Please upload a MP4 video.")
            
            if not self.model:
                raise ValueError("Model not loaded properly.")
            
            # Stores the uploaded file
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.uploads_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(file.file.read())

            # Process the video
            cap = cv2.VideoCapture(str(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.outputs_dir / f"processed_5_{file.filename}"

            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
            count = 1
            while cap.isOpened():
                
                ret, frame = cap.read()
                
                
                if not ret:
                    break

                print(f"frame {count} and {frame.shape}")
                count += 1

                results = self.model(frame, conf=0.25, imgsz=1280)

                detections = []

                for r in results:
                    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                        x1, y1, x2, y2 = map(int, box)
                        detections.append([x1, y1, x2, y2, float(conf)])

                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        # cv2.putText(frame, f"{int(cls)} {conf:.2f}", (x1, y1-10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                tracks = self.tracker.update_tracks(detections, frame.shape[:2])

                # Update line counter with tracks
                counter_result = self.line_counter.update(tracks, frame.shape[:2])

                # Draw tracking boxes and IDs
                for x1, y1, x2, y2, track_id in tracks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{int(track_id)}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Draw counting line and statistics with improved visualization
                self.line_counter.draw_line_and_counts(frame, show_direction_arrows=True)
                self.line_counter.draw_tracking_points(frame, counter_result['current_positions'], show_trails=True)

                out.write(frame)

            cap.release()
            out.release()

            # Get final counting stats
            final_stats = self.line_counter.get_stats()

            # Reset services for next video
            self.tracker.reset()
            self.line_counter.reset()

            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "output_path": str(output_path),
                "counting_stats": {
                    "in_count": final_stats['in_count'],
                    "out_count": final_stats['out_count'],
                    "total_count": final_stats['total_count']
                }
            }
        
        except Exception as e:
            return {"error": str(e)}