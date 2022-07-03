from flask import Flask, redirect, request, render_template, flash, url_for
import logging
from logging.handlers import RotatingFileHandler
from isort import file
import pandas as pd
pd.set_option('display.float_format', '{:.3f}'.format)
from load_the_model import load_model
import numpy as np
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import os

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
# ERROR LOGGER:
def init_error_logger():
    logger = logging.getLogger('IMAGE CLASSIFICATION ERRORS')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level=logging.DEBUG)
    # add a log rotating handler (rotates when the file becomes 10MB, or about 100k lines):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler('logs/server_error.log', maxBytes=10000000, backupCount=10)
    handler.setLevel(level=logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger




##############################################
# Variables
##############################################
app.logger = init_logger()
app.error_logger = init_error_logger()
app.secret_key = "vnimanieAI"
app.categories = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry',
                  'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya',
                  'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']
# Predictor model
app.predictor = load_model(last_layer_file_name = 'trained_layer_params.pth', num_categories = len(app.categories)) # Default argument for file name
app.category_dictionary = {}
for i, ele in enumerate(app.categories):
    app.category_dictionary.update({i : ele})
app.ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER





########################################
# FUNCTIONS
########################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.ALLOWED_EXTENSIONS


def predict(image_path):
    try:
        transformer = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image = Image.open(image_path)
        to_predict = transformer(image)
        pred_tensor = app.predictor(to_predict.unsqueeze(0))
        predicted_category = pred_tensor.argmax().item()
        final_prediction = app.category_dictionary[predicted_category]
        return final_prediction
    except Exception as e:
        return f"Model couldn't predict! Reason: [{e}]"






####################################################
# APP ROUTES
####################################################
@app.route('/', methods = ['GET', 'POST'])
def home():

    title = "Image classificator"
    if 'file' not in request.files:
        flash('No file part')
        return render_template('index.html', title = title)
    image = request.files['file']
    if image.filename == '':
        flash('No image selected for predicting')
        return render_template('index.html', title = title)
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash("Image succesfully uploaded! Initiating prediction...\n")
        try:
            result = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash("According to the neural network this is "+result)
        except Exception as e:
            flash(f"Neural Network couldn't process the image because of [{e}]")
        return render_template('index.html', title = title, filename = filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return render_template('index.html', title = title)

@app.route('/prediction/<filename>')
def display_prediction(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code = 301)
    

if __name__ == "__main__":
    app.run()





























