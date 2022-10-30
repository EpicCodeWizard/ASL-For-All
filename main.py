from flask_socketio import SocketIO, emit
from fix_cors import fix_cors
import tensorflow as tf
from flask import *
import numpy as np
import PIL

app = Flask(__name__, template_folder="", static_folder="")
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")
model = tf.keras.models.load_model(r"C:\Users\Sarvesh\Downloads\wdefrgftg\model.h5")
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def parse(bytecode):
  img1 = PIL.JpegImagePlugin.JpegImageFile(io.BytesIO(bytecode))
  img_array = tf.keras.utils.img_to_array(img1)
  img_array = tf.expand_dims(img_array, 0)
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  return class_names[np.argmax(score)], 100 * np.max(score)

@app.route("/")
@fix_cors
def main():
  return render_template("index.html")

@socketio.on("img")
def imghandle(img):
  emit("txt", parse(img))

socketio.run(app, host="0.0.0.0")
