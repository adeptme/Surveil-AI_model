import cv2
from ultralytics import YOLO
import csv
import time
from datetime import datetime
import os

MODEL_PATH = "runs/detect/train/weights/best.pt" 

# VIDEO_SOURCE = 0  # Change to video path like "path/to/video.mp4"
VIDEO_SOURCE = "testvid-1.mp4"

# Recording interval in seconds (how often to save count to CSV)
RECORD_INTERVAL = 5

PROCESS_FPS = 15  # Set to a number 10, 15, 30 

CSV_OUTPUT = "vehicle_counts.csv"

CONFIDENCE_THRESHOLD = 0.4

VEHICLE_CLASSES = ['bus', 'car', 'jeepney', 'motorcycle', 'pickup-truck', 'truck', 'van']

def initialize_csv(csv_file):
    """Initialize CSV file with headers if it doesn't exist"""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Elapsed_Time_Seconds', 'Vehicle_Count', 
                           'Buses', 'Cars', 'Jeepneys', 'Motorcycles', 'Pickup_Trucks', 'Trucks', 'Vans'])


def count_vehicles(results, class_names):
    """Count detected vehicles by type"""
    counts = {
        'total': 0,
        'bus': 0,
        'car': 0,
        'jeepney': 0,
        'motorcycle': 0,
        'pickup-truck': 0,
        'truck': 0,
        'van': 0
    }
    
    # Get detections from results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id].lower()
            
            if conf >= CONFIDENCE_THRESHOLD and cls_name in VEHICLE_CLASSES:
                counts['total'] += 1
                if cls_name in counts:
                    counts[cls_name] += 1
    
    return counts


def save_to_csv(csv_file, timestamp, elapsed_time, counts):
    """Save vehicle count data to CSV"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            f"{elapsed_time:.2f}",
            counts['total'],
            counts['bus'],
            counts['car'],
            counts['jeepney'],
            counts['motorcycle'],
            counts['pickup-truck'],
            counts['truck'],
            counts['van']
        ])


def main():
    print("=" * 60)
    print("Vehicle Detection and Counting System")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Video Source: {VIDEO_SOURCE}")
    print(f"Recording Interval: {RECORD_INTERVAL} seconds")
    print(f"CSV Output: {CSV_OUTPUT}")
    print("=" * 60)
    print("\nPress 'q' to quit\n")
    
    # Load YOLO model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Model loaded successfully! Classes: {len(class_names)}")
    
    # Initialize CSV
    initialize_csv(CSV_OUTPUT)
    
    # Open video source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps")
    
    # Calculate frame skip interval
    if PROCESS_FPS is not None and PROCESS_FPS > 0:
        frame_skip = max(1, int(fps / PROCESS_FPS))
        actual_process_fps = fps / frame_skip
        print(f"Processing every {frame_skip} frame(s) (~{actual_process_fps:.1f} fps)")
    else:
        frame_skip = 1
        print(f"Processing all frames ({fps} fps)")
    
    # Timing variables
    start_time = time.time()
    last_record_time = start_time
    frame_count = 0
    
    print("\nStarting detection...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        frame_count += 1
        
        # Skip frames based on PROCESS_FPS setting
        if frame_count % frame_skip != 0:
            continue
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Run inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Count vehicles
        counts = count_vehicles(results, class_names)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        
        # Add information overlay
        info_y = 30
        cv2.putText(annotated_frame, f"Time Elapsed: {elapsed_time:.2f}s", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(annotated_frame, f"Total Vehicles: {counts['total']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show breakdown by type
        info_y += 30
        breakdown = f"Cars: {counts['car']} | Jeepneys: {counts['jeepney']} | Buses: {counts['bus']} | Vans: {counts['van']}"
        cv2.putText(annotated_frame, breakdown, 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        info_y += 25
        breakdown2 = f"Trucks: {counts['truck']} | Pickups: {counts['pickup-truck']} | Motorcycles: {counts['motorcycle']}"
        cv2.putText(annotated_frame, breakdown2, 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Check if it's time to record
        if current_time - last_record_time >= RECORD_INTERVAL:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_to_csv(CSV_OUTPUT, timestamp, elapsed_time, counts)
            last_record_time = current_time
            print(f"[{timestamp}] Elapsed: {elapsed_time:.2f}s | Vehicles: {counts['total']} | "
                  f"Cars: {counts['car']}, Jeepneys: {counts['jeepney']}, Buses: {counts['bus']}, "
                  f"Vans: {counts['van']}, Trucks: {counts['truck']}, Pickups: {counts['pickup-truck']}, "
                  f"Motorcycles: {counts['motorcycle']}")
        
        # Display the frame
        cv2.imshow('Vehicle Detection', annotated_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping detection...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Session Summary:")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Results saved to: {CSV_OUTPUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
