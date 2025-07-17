from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload and result folders inside 'static'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your trained model once
model = YOLO("yolov11_custom.pt")

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file part exists
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Run detection
            results = model.predict(source=input_path, conf=0.6, save=True, save_dir=RESULT_FOLDER)

            # results[0].path contains path to output file, get relative path from 'static/'
            output_path = os.path.relpath(results[0].path, 'static')

            # Relative input path from static for rendering
            input_rel_path = os.path.relpath(input_path, 'static')

            # If it's an image show result_image.html, if video result_video.html
            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['mp4', 'mov', 'avi']:
                return render_template("result_video.html", input_video=input_rel_path, output_video=output_path)
            else:
                return render_template("result_image.html", input_image=input_rel_path, output_image=output_path)
        else:
            return "File type not allowed", 400

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
