
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw
import cv2
import os, time, uuid

# Replace with valid custom vision api key values
TRAINENDPOINT = ""
ENDPOINT = ""
training_key = ""
prediction_key = ""
prediction_resource_id = ""

#create client objects with the api keys
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

all_projects = trainer.get_projects()
mine_project = all_projects[0]

# Now there is a trained endpoint that can be used to make a prediction

publish_iteration_name = "Iteration4"
base_image_location = ""

#minimum prediction precision to output marking box
prediction_threshold = 50

# Open the sample image and get back the prediction results.
with open(os.path.join (base_image_location, ""), mode="rb") as test_data:
    results = predictor.detect_image(mine_project.id, publish_iteration_name, test_data)

img = Image.open(base_image_location + "/")
# Display the results.    
for prediction in results.predictions:
    predict_left = prediction.bounding_box.left
    predict_top = prediction.bounding_box.top
    predict_width = prediction.bounding_box.width
    predict_height = prediction.bounding_box.height
    predict_prob = prediction.probability * 100
    if(predict_prob >= prediction_threshold):
        print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(predict_prob, predict_left, predict_top, predict_width, predict_height))

        box_width = float(predict_width) * float(img.width)
        box_heigth = float(predict_height) * float(img.height)

        box_x1 = float(predict_left) * float(img.width)
        box_y1 = float(predict_top) * float(img.height)
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_heigth

        draw = ImageDraw.Draw(img)
        draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)],outline="red", width=10)
        img.save(base_image_location + "/")

img.show()

