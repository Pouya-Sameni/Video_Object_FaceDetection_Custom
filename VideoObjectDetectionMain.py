from ntpath import join
from imageai.Detection import VideoObjectDetection
from imageai.Detection.Custom import CustomVideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt

def outputFrame (frame_number, output_array, output_count, returned_frame):
    plt.clf()
    #plt.subplot(1, 2, 1)
    plt.title("Frame : " + str(frame_number))
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")
    plt.pause(0.001)
executionPath = os.getcwd()

camera = cv2.VideoCapture(0)


detector = CustomVideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setJsonPath("detection_config.json")
detector.setModelPath(os.path.join(executionPath, "FaceDetectionModel V2.h5"))
detector.loadModel()

plt.show()

detections = detector.detectObjectsFromVideo(
    camera_input=camera,
    output_file_path=os.path.join(executionPath, "OutputVideo"), frames_per_second=24, log_progress=True, per_frame_function=outputFrame,
    minimum_percentage_probability=20, return_detected_frame=True
    
    )

# for item in detections:
#     print (item["name"],
#     " : ",
#     item["percentage_probability"])




