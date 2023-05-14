import os
import cv2
import numpy as np
from io import BytesIO
from flask import Flask, request, send_file, send_from_directory
import mediapipe as mp

app = Flask(__name__)
app.secret_key = b'\xec\xe1\xa5nv\x17>\x97\x89%l\xfd\x83\xbc\\z'

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)


def overlay_product(person_img, product_img):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(
            cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = person_img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin *
                                                       ih), int(bboxC.width * iw), int(bboxC.height * ih)

                top = y - int(h * 0.8)
                bottom = y + h + int(h * 0.12)
                left = x - int(w * 0.12)
                right = x + w + int(w * 0.12)

                width = right - left
                height = bottom - top

                product_resized = cv2.resize(product_img, (width, height))

                if product_resized.shape[2] == 4:
                    alpha = product_resized[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)

                    if person_img[top:top + height, left:left + width].shape[:2] == product_resized[:, :, :3].shape[:2]:
                        person_img[top:top + height, left:left + width] = (
                            1 - alpha) * person_img[top:top + height, left:left + width] + alpha * product_resized[:, :, :3]
                else:
                    if person_img[top:top + height, left:left + width].shape == product_resized.shape:
                        person_img[top:top + height,
                                   left:left + width] = product_resized

    return person_img


@app.route('/overlay', methods=['POST'])
def overlay():
    person_img = request.files['person']
    product_img = request.files['product']

    person_img = cv2.imdecode(np.frombuffer(
        person_img.read(), np.uint8), cv2.IMREAD_COLOR)
    product_img = cv2.imdecode(np.frombuffer(
        product_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    result = overlay_product(person_img, product_img)

    buffer = BytesIO()
    _, encoded = cv2.imencode('.jpg', result)
    buffer.write(encoded)

    buffer.seek(0)

    return send_file(buffer, mimetype='image/jpeg')


@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)


@app.route('/')
def index():
    return send_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)
