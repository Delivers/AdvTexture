import cv2
import numpy as np
import time

# Load YOLOv2
net_v2 = cv2.dnn.readNet("yolov2.weights", "yolov2.cfg")
layer_names_v2 = net_v2.getLayerNames()
output_layers_v2 = [layer_names_v2[i - 1] for i in net_v2.getUnconnectedOutLayers().flatten()]

# Load YOLO-Lite (YOLOv3-tiny)
net_lite = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names_lite = net_lite.getLayerNames()
output_layers_lite = [layer_names_lite[i - 1] for i in net_lite.getUnconnectedOutLayers().flatten()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set preferred backend and target for both networks
net_v2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_v2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net_lite.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_lite.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Initialize variables for FPS calculation
fps_start_time = 0
fps = 0
use_yolo_lite = True  # Set to False to use YOLOv2



while True:
    fps_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    # Resize the frame to full screen size
    #frame = cv2.resize(frame, (2160, 3840))

    # Stretch the frame horizontally by a factor
    stretch_factor = 1  # Adjust this factor as needed
    stretched_width = int(frame.shape[1] * stretch_factor)
    frame = cv2.resize(frame, (stretched_width, frame.shape[0]))

     # Rotate the frame 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (312, 312), (0, 0, 0), True, crop=False)
    
    if use_yolo_lite:
        net_lite.setInput(blob)
        outs = net_lite.forward(output_layers_lite)
    else:
        net_v2.setInput(blob)
        outs = net_v2.forward(output_layers_v2)

    # Show information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Color for the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate and display FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Create a fullscreen window
    cv2.namedWindow("YOLO Object Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLO Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("YOLO Object Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to switch between YOLOv2 and YOLO-Lite
        use_yolo_lite = not use_yolo_lite
    if key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
