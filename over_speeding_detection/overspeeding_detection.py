import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- SETTINGS ---
video_path = r'video_samples\2165-155327596_small.mp4' 
scale_factor = 0.05  # Meters per pixel (calibrate for real-world accuracy)
speed_limit = 60  # Speed limit in km/h for overspeeding alert

# --- INIT MODELS ---
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()
print("Video opened successfully!")

# --- GET FPS ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30  # Fallback value
print(f"Video FPS: {fps}")

# --- TRACKING ---
prev_positions = {}

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    # --- YOLO DETECTION ---
    results = model(frame, verbose=False)[0]
    detections = []

    # Filter vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        if int(cls) in [2, 3, 5, 7]:
            detections.append(([x1, y1, x2, y2], conf, 'vehicle'))

    # --- TRACK OBJECTS ---
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx, cy = (l + r) // 2, (t + b) // 2
        curr_pos = (cx, cy)

        # --- SPEED CALCULATION ---
        if track_id in prev_positions:
            dx = curr_pos[0] - prev_positions[track_id][0]
            dy = curr_pos[1] - prev_positions[track_id][1]
            pixel_distance = np.sqrt(dx**2 + dy**2)
            meter_distance = pixel_distance * scale_factor
            speed_mps = meter_distance * fps
            speed_kph = speed_mps * 3.6

            # --- PRINT TO TERMINAL ---
            print(f"Track ID: {track_id} | Speed: {speed_kph:.2f} km/h | "
                  f"Pixel Distance: {pixel_distance:.2f} px | FPS: {fps:.2f}")

            # --- OVERLAY ON VIDEO ---
            cv2.putText(frame, f"{speed_kph:.1f} km/h", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if speed_kph > speed_limit:
                cv2.putText(frame, "Over-Speeding!", (int(l), int(b) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update previous position
        prev_positions[track_id] = curr_pos

        # --- DRAW BOUNDING BOX & ID ---
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- DISPLAY FRAME ---
    cv2.imshow("Over-Speeding Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Exit requested by user.")
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Program finished.")


