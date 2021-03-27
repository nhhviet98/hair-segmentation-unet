from flask import request, send_from_directory
import cv2
import numpy as np
from flask_api import FlaskAPI, status, exceptions
from Predict import predict

app = FlaskAPI(__name__, static_url_path="")


@app.errorhandler(400)
def handle_400(_err):
    return status.HTTP_400_BAD_REQUEST


@app.errorhandler(401)
def handle_401(_err):
    return status.HTTP_401_UNAUTHORIZED


@app.errorhandler(404)
def handle_404(_err):
    return status.HTTP_404_NOT_FOUND


@app.errorhandler(500)
def handle_500(_err):
    return status.HTTP_500_INTERNAL_SERVER_ERROR


@app.route("/hair-segmentation", methods=['POST'])
def solve_expression():
    if request.method == "POST":
        try:
            file = request.files["image"]
            img_byte = file.read()
            img = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)
            mask = predict(img)
            mask_path = "results/mask" + file.filename
            cv2.imwrite(mask_path, mask*255)
            return {
                "filename": mask_path
            }
        except:
            handle_404()


@app.route('/file/<path:path>')
def static_file(path):
    return send_from_directory("", path)


def main():
    port = 5000
    app.run(debug=True, port=port, host="0.0.0.0")


if __name__ == '__main__':
    main()




