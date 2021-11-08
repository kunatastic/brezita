from flask import Flask,render_template,request
from PIL import Image
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import keras
from keras.models import model_from_json

app = Flask(__name__)

@app.route("/")
def hello_world():
  return render_template('index.html')


def init():
  json_file = open('./models/model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights("./models/model.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.get_default_graph()
  return loaded_model,graph

# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()


def predict_image(img):
  info = {1: '3', 5: '7', 17: 'x', 0: '2', 16: 'w', 8: 'c', 6: '8', 15: 'p', 4: '6', 14: 'n', 2: '4', 7: 'b', 13: 'm', 3: '5', 11: 'f', 9: 'd', 18: 'y', 10: 'e', 12: 'g'}
  img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
  img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))
  img = cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)
  img = cv2.GaussianBlur(img, (1,1), 0)

  image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
  
  Xdemo = []
  for i in range(5):
    Xdemo.append(img_to_array(image_list[i]))
  Xdemo = np.array(Xdemo)
  Xdemo/=255

  ydemo = model.predict(Xdemo)
  ydemo = np.argmax(ydemo, axis=1)

  result = ""
  for res in ydemo:
    result += info[res]
  return result




@app.route("/predict/",methods=["POST"])
def predict():
  # img = Request.args.get('img')

  img = request.files['img']
  print(img)
  return predict_image(img)

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)