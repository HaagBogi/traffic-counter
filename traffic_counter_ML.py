import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pandas as pd
from tqdm import tqdm
import os
import maximum_matching as mm
import math

cap = cv2.VideoCapture(os.path.join("data", "traffic_video.avi"))
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# creates a pandas data frame with the number of rows the same length as frame count
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"


frames_count = int(frames_count)

OPTIMAL = 45

START = 40
END = 101
best_count = -1
for threshold in range(START, END):
    car_count = 0
    print("THRESHOLD: ", threshold)
    STOP_EXP = False
    for i in tqdm(range(frames_count), total=frames_count, desc="Processing frames..."):
        if (i % 60 == 0) and i != 0:
            print(f"Found {car_count} cars so far.")

        ret, frame = cap.read()  # import image
        if ret:
            bbox, label, conf = cv.detect_common_objects(frame)
            central_positions = []
            for j in range(len(bbox)):
                if label[j] == "car":
                    # x, y starts from top left
                    topleft_x, topleft_y, width, height = bbox[j]
                    central_x = int(topleft_x + (width/2))
                    central_y = int(topleft_y + (height/2)) # Y going downwards is +
                    central_positions.append((central_x, central_y))

            if i >= 1:
                number_of_new_cars = len(central_positions) - mm.maximum_matching(central_positions, prev_central_position, threshold)
                car_count += number_of_new_cars
            else:
                car_count += len(central_positions)

            prev_central_position = central_positions

        if abs(OPTIMAL - car_count) > 100:
            STOP_EXP = True
            break

    cap.release()
    cap = cv2.VideoCapture(os.path.join("data", "traffic_video.avi"))
    if STOP_EXP:
        STOP_EXP = False
        car_count = 0
        continue

    if best_count == -1:
        best_count = car_count
        best_diff = abs(OPTIMAL - car_count)
        best_thresh = threshold
    else:
        if abs(OPTIMAL - car_count) < best_diff:
            best_diff = abs(OPTIMAL - car_count)
            best_count = car_count
            best_thresh = threshold


if best_count == -1:
    print("Experiment failed")
    exit()

print("Best total car count: ", best_count)
print("Best threshold: ", best_thresh)