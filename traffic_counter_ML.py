import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pandas as pd
from tqdm import tqdm

cap = cv2.VideoCapture('traffic_video.avi')
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = int(width)
height = int(height)
print(frames_count, fps, width, height)

# creates a pandas data frame with the number of rows the same length as frame count
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name = "Frames"


frames_count = int(frames_count)
car_count = 0

for i in tqdm(range(frames_count), total=frames_count, desc="Processing frames..."):
    if (i % 60 == 0) and i != 0:
        print(f"Found {car_count} cars so far.")

    ret, frame = cap.read()  # import image
    if ret:
        bbox, label, conf = cv.detect_common_objects(frame)
        cars_on_frame = label.count('car')
        if car_count == 0:
            car_count = cars_on_frame
            cars_on_prev_frame = cars_on_frame
        else:
            if cars_on_frame > cars_on_prev_frame:
                car_count += (cars_on_frame - cars_on_prev_frame)
                cars_on_prev_frame = cars_on_frame

print("Total car count: ", car_count)