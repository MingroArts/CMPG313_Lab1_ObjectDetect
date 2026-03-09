from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

# Set source: 0 for webcam, or "filename.jpg" / "filename.mp4"
source = 0 

# Check if source is text (a file) or a number (webcam)
if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    model.predict(source=source, conf=0.35, show=True, save=True)
else:
    # This will now handle the webcam (0) or video files correctly
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)

print("Done. Check runs/ folder.")