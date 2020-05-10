import cv2
import numpy as np
import time
import itertools
from scipy.spatial import distance

# Load Yolo
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-model/yolov3.cfg", "yolo-model/yolov3.weights")

# # with GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
outputFile = "output.avi"

with open("yolo-model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def main():
    # Loading camera
    cap = cv2.VideoCapture("mall.mp4")

    # video writer
    writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    font = cv2.FONT_HERSHEY_DUPLEX
    starting_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        frame_id += 1

        height, width, channels = frame.shape[:3]

        if not ret:
            break

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        centers = []
        MIN_DIST = 150
        mindistances = {}

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != 0:  # filter person class
                    continue
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
                    centers.append((center_x, center_y))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y - 5), font, 1, (0,255,0), 1)
        
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 0, 0), 1)

        def find_min_distance(centers):
            '''
            return min eculidian distance between predicted anchor boxes
            '''
            comp = list(itertools.combinations(centers, 2))
            for pts in comp:
                ecdist = np.linalg.norm(np.asarray(pts[0])-np.asarray(pts[1]))
                if ecdist < MIN_DIST:
                    mindistances.update({pts: ecdist})
        
        find_min_distance(centers)
        for d in mindistances:
            cv2.arrowedLine(frame, d[0], d[1], (0, 0, 255), 3)
        
        writer.write(frame.astype(np.uint8))
        cv2.namedWindow("Social Distancing", cv2.WINDOW_NORMAL)
        cv2.imshow("Social Distancing", frame)
        
        # press ESC to exit
        key = cv2.waitKey(1)
        if key == 27:
            break

    print("Done!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()