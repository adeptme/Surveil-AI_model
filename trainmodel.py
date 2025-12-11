from ultralytics import YOLO

# fixes RuntimeError where an attempt ... bootstrapping phase.

if __name__ == '__main__':
    # model = YOLO("yolo11n.pt")
    # model = YOLO("runs\\detect\\train2\\weights\\best.pt") # go yolo11s-v4dataset-train2-last.pt if continuing training

    # Train the model
    # results = model.train(data="data.yaml", epochs=60, imgsz=640, batch=8)

    # Run inference with the YOLO11s model on the 'bus.jpg' image batch=8
    # For testing model


    # Evaluate the model's performance on the validation set
    best = YOLO("runs/detect/train/weights/best.pt")
    # results = best("", save=True, show=True)
    results = best.val(data="dataset/data.yaml", save=True)
    # print(results.box.map)

# Export the model to ONNX format
# success = model.export(format="onnx")
