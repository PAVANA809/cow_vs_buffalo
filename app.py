from flask import json, request,make_response,render_template,redirect,url_for
from flask import jsonify
from flask import Flask
import os
import base64
from PIL import Image
import io
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

app.config["IMAGE_UPLOADS"] = "static/images/uploads"

@app.route('/upload_image',methods=['POST','GET'])
def upload_image():
    print("before post")
    if request.method == "POST":
        print("after post")
        message = request.get_json(force=True)
        encoded = message['image']
        image_data = re.sub('^data:image/.+;base64,', '', encoded)
        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
        print("image decoded")
        # web_img = request.files["file"]
        # image.save(os.path.join(app.config["IMAGE_UPLOADS"], Image.filename))
        # print("Image is saved")
        # path = "static/images/uploads/"+image.filename
        # img = image.load_img(path)
        print("loading model")
        get_model()
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = predict(processed_image)
        print(prediction)
        response = {
                'cow': prediction[0][0],
                'buffalo': prediction[0][1]
            }
        print(response)
        return jsonify(response)
    return



def get_model():
    global model
    model = load_model('mobile_net_after_training_cowvsbuffalo.h5')
    print(" * Model loaded!")


def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    global pre
    pre = model.predict(img).tolist()
    return pre

if __name__ == '__main__':
    app.run(debug=True)