from ultralytics import YOLO
from multiprocessing import Process, freeze_support, set_start_method

# fixes RuntimeError where an attempt ... bootstrapping phase.

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    # Train the model 
    results = model.train(data="data.yaml", epochs=60, imgsz=[640, 480])

    # Run inference with the YOLO11s model on the 'bus.jpg' image
    # For testing model
    

    # Evaluate the model's performance on the validation set
    best = YOLO("runs/detect/train/weights/best.pt")
    results = best("testvideo.mp4", save=True, show=True)
    results = best.val()
    print(results.box.map)

# Export the model to ONNX format
# success = model.export(format="onnx")
