import cv2
import time
from ultralytics import FastSAM

def main():
    # 1. Load the small FastSAM model (downloads ~45MB automatically on first run)
    print("Loading FastSAM model...")
    model = FastSAM("FastSAM-s.pt")

    # 2. Open default laptop webcam (index 0)
    cap = cv2.VideoCapture(1)

    # Set camera resolution (640x480 keeps CPU inference responsive)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open laptop webcam.")
        return

    print("Webcam started. Press 'q' to exit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Run class-agnostic segmentation
        # device='cpu' for now; retina_masks=False keeps processing fast
        results = model(
            frame,
            device='cpu',
            retina_masks=False,
            imgsz=480,
            conf=0.3,       # Confidence threshold for shapes/borders
            iou=0.7,        # Overlap threshold to eliminate duplicate boxes
            verbose=False
        )

        annotated_frame = frame.copy()

        # 4. Extract and draw bounding boxes around every detected shape
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates: top-left (x1, y1) and bottom-right (x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Draw a green bounding rectangle around each detected shape
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5. Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # 6. Show the live feed
        cv2.imshow("Shape & Object Bounding (Laptop Webcam)", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()