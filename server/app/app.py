from flask import Flask, request, render_template
import logging
from logging.handlers import RotatingFileHandler
import time, atexit
import pandas as pd
pd.set_option('display.float_format', '{:.3f}'.format)
from load_the_model import load_model
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image

app = Flask(__name__)

# LOGGER:
def init_logger():
    logger = logging.getLogger('IMAGE CLASSIFICATION')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level=logging.DEBUG)
    # add a log rotating handler (rotates when the file becomes 10MB, or about 100k lines):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler('logs/server.log', maxBytes=10000000, backupCount=10)
    handler.setLevel(level=logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger





##############################################
# Variables
##############################################
app.logger = init_logger()
app.categories = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry',
                  'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya',
                  'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']
# Predictor model
app.predictor = load_model(file_name = 'trained_layer_params.pth', num_categories = len(app.categories)) # Default argument for file name
app.category_dictionary = {}
for i, ele in enumerate(app.categories):
    app.category_dictionary.update({i : ele})






########################################
# FUNCTIONS
########################################
def predict(image):
    try:
        transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        to_predict = transformer(image)
        pred_tensor = app.predictor(to_predict.unsqueeze(0))
        predicted_category = pred_tensor.argmax()
        final_prediction = app.category_dictionary(predicted_category)
        return final_prediction
    except Exception as e:
        return f"Model couldn't predict! [{e}]"






####################################################
# APP ROUTES
####################################################
@app.route('/home', methods = ['GET', 'POST'])
def home():

    title = "Image classificator"






























