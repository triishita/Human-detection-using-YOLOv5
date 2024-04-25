import torch
import cv2

# Load YOLOv5 model
#model = torch.hub.load('yolov5', 'yolov5s', source="local")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
video_path = 'yolo/reference.mp4'
cap = cv2.VideoCapture(video_path)

# Set reference distance in meters (you need to measure this in your scene)
reference_distance_meters = 3.0 # Example: 3 meters

# Average human height in meters
average_human_height_meters = 1.7 # Example: 1.7 meters

# Dictionary to store heights of each person
person_heights = {}
person_face_positions = {}

# Initialize face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Filter detections for 'person' class
    person_results = results.xyxy[0][results.xyxy[0][:, -1] == 0]

    # Calculate scale factor (pixels per meter)
    scale_factor = average_human_height_meters / person_results[:, 3].mean()

    # Draw bounding boxes for persons and calculate height
    for xmin, ymin, xmax, ymax, conf, cls in person_results:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate height in meters
        height_meters = (ymax - ymin) * scale_factor

        # Detect faces in the bounding box
        faces = face_detector.detectMultiScale(frame[int(ymin):int(ymax), int(xmin):int(xmax)], scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Track person based on face position
            face_x, face_y, face_w, face_h = faces[0]  # Take the first detected face
            face_center = (int(xmin + face_x + face_w / 2), int(ymin + face_y + face_h / 2))

            if face_center not in person_face_positions:
                # New person detected
                person_id = f'Person {len(person_heights) + 1}'
                person_heights[person_id] = height_meters
                person_face_positions[face_center] = person_id
            else:
                # Update existing person's height
                person_id = person_face_positions[face_center]
                person_heights[person_id] = max(person_heights[person_id], height_meters)

        cv2.putText(frame, f'Height: {height_meters:.2f} m', (int(xmin), int(ymin - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the highest height for each person
for person_id, height in person_heights.items():
    print(f'{person_id}: {height:.2f} m')