from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import os

from werkzeug.utils import redirect

app = Flask(__name__)
Upload = 'static\\storage'
app.config['uploadFolder'] = Upload
ALLOWED_EXTENSIONS = set(['png'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def init():
    # json_file = open('./models/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # #load weights into new model
    # loaded_model.load_weights("./models/model.h5")
    loaded_model = load_model("./models/solver-model.h5")
    print("Loaded Model from disk")
    return loaded_model


# global vars for easy reusability
global model, graph
# initialize these variables
model = init()


@app.route("/")
def hello_world():
    return render_template('index.html')


def predict_captcha(img):
    info = {
        0: '2',
        1: '3',
        2: '4',
        3: '5',
        4: '6',
        5: '7',
        6: '8',
        7: 'b',
        8: 'c',
        9: 'd',
        10: 'e',
        11: 'f',
        12: 'g',
        13: 'm',
        14: 'n',
        15: 'p',
        16: 'w',
        17: 'x',
        18: 'y',
    }

    # img = load_img(img)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 145, 0)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))
    img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)
    img = cv2.GaussianBlur(img, (1, 1), 0)

    image_list = [
        img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90],
        img[10:50, 90:110], img[10:50, 110:130]
    ]

    Xdemo = []
    for i in range(5):
        Xdemo.append(img_to_array(image_list[i]))
    Xdemo = np.array(Xdemo)
    Xdemo /= 255

    ydemo = model.predict(Xdemo)
    ydemo = np.argmax(ydemo, axis=1)

    result = ""
    for res in ydemo:
        result += info[res]

    print(result)
    return result


@app.route("/predict/", methods=["GET", "POST"])
def predict():

    if (request.method == "POST"):
        try:
            print("HEY BRO I WAS CALLED")
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['uploadFolder'],
                                        file.filename)
                file.save(filename)
                result = predict_captcha(filename)
                return render_template("result.html",
                                       result=result,
                                       captcha_img=filename)

        except Exception as e:
            print(e)
            return "Error"
    elif (request.method == "GET"):
        return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
