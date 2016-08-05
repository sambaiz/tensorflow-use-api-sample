from flask import Flask, request, jsonify
import tensorflow as tf
from mnist import Mnist

app = Flask(__name__)

mnist = Mnist()
mnist.restore("/model.ckpt")

@app.route("/", methods=['POST'])
def what_number():

    json = request.json
    if(json is None or "image" not in json or len(json["image"]) != 784):
        return jsonify(error="Need json includes image property which is 784(28 * 28) length, float([0, 1.0]) array")
    else:
        result = list(mnist.what_number([json["image"]]))
        return jsonify(result=result[0])

if __name__ == "__main__":
    app.run(port=3000, host='0.0.0.0')
