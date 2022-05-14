from imageai.Detection.Custom import DetectionModelTrainer
import tensorflow as tf
# model_trainer = ClassificationModelTrainer()

# model_trainer.setModelTypeAsResNet()

# model_trainer.setDataDirectory('C:/Users/pouya/Desktop/Learning Python/Test Projects/ObjectDetectionVideo/Pouya_Sameni_Frames')

# model_trainer.trainModel(num_objects=1, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)



trainer = DetectionModelTrainer()

trainer.setModelTypeAsYOLOv3()

trainer.setDataDirectory(data_directory="Pouya_Face_Model")

trainer.setTrainConfig(object_names_array=["Pouya"], batch_size=5, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")

#trainer.setTrainConfig(object_names_array=["Pouya"], batch_size=5, num_experiments=100)


trainer.trainModel()