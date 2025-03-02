import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker  # Ensure you have the Tracker class implemented
import time

import ultralytics
ultralytics.__version__

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Define class list
class_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light'
]

# Initialize video capture
cap = cv2.VideoCapture('highway.mp4')  # Update the path to your video file

# Initialize variables
count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

# Define line positions and offset
red_line_y = 198
blue_line_y = 268
offset = 6

# # Create a folder to save frames
# if not os.path.exists('detected_frames'):
#     os.makedirs('detected_frames')

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    # Predict objects in the frame
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []

    # Extract bounding boxes for cars
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    # Update tracker with detected cars
    bbox_id = tracker.update(list)

    # Process each tracked object
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Check if vehicle crosses the red line (going down)
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()  # Record time when vehicle touches the red line
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]  # Calculate elapsed time
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10  # Distance between lines in meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6  # Convert to km/h
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Check if vehicle crosses the blue line (going up)
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()  # Record time when vehicle touches the blue line
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = time.time() - up[id]  # Calculate elapsed time
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance1 = 10  # Distance between lines in meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6  # Convert to km/h
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw lines and text on the frame
    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, 'Red Line', (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
    cv2.putText(frame, 'Blue Line', (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Going Down - ' + str(len(counter_down)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Going Up - ' + str(len(counter_up)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # # Save frame
    # frame_filename = f'detected_frames/frame_{count}.jpg'
    # cv2.imwrite(frame_filename, frame)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()