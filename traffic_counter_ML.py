import cvlib as cv

def detect_cars_on_frame(frame):
    bbox, label, conf = cv.detect_common_objects(frame, model="yolov4")
    bbox2, label2, conf2 = [], [], []
    for i in range(len(label)):
        if label[i] == "car" or label[i] == "motorcycle" or label[i] == "truck" or label[i] == "bus":
            bbox2.append(bbox[i])
            label2.append(label[i])
            conf2.append(conf[i])

    return bbox2, label2, conf2
