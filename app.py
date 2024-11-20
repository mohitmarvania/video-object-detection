from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import shutil
import zipfile
import ffmpeg
import imageio_ffmpeg as iio
import cv2
from ultralytics import YOLO
import csv
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FRAME_FOLDER'] = 'frames'
app.config['PROCESSED_FRAME_FOLDER'] = 'processed_frames'
app.config['OUTPUT_FOLDER'] = 'output'

model = YOLO('yolo11n.pt')  # Loading YOLO models

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['FRAME_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['PROCESSED_FRAME_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_video():
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    # Extract frames
    extract_frames(video_path, app.config['FRAME_FOLDER'])
    return render_template('results.html')


def extract_frames(videoPath, output_dir, frame_rate=1):
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now()

    ffmpeg_binary = iio.get_ffmpeg_exe()
    (
        ffmpeg.input(videoPath)
            .output(
            f'{output_dir}/frame_%04d.jpg',
            vf=f'fps={frame_rate}')
            .run(cmd=ffmpeg_binary)
    )

    process_frames()
    # ffmpeg.input(videoPath).output(f'{output_dir}/frame{current_time.day}-{current_time.month}-{
    # current_time.year}_{current_time.hour}:{current_time.minute}_%04d.jpg', vf=f'fps={frame_rate}').run()


@app.route('/process', methods=['POST', 'GET'])
def process_frames():
    frames_dir = app.config['FRAME_FOLDER']
    output_dir = app.config['PROCESSED_FRAME_FOLDER']
    csv_dir = app.config['OUTPUT_FOLDER']
    current_time = datetime.datetime.now()
    csv_file = os.path.join(csv_dir,
                            f'objects_count_{current_time.day}-{current_time.month}-{current_time.year}.csv')

    detect_objects(frames_dir, output_dir, csv_file)

    processed_frames = os.listdir(output_dir)
    return render_template('results.html', frames=processed_frames, csv_file=csv_file)


def detect_objects(frames_dir, output_dir, csv_file):
    frame_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Object', 'Count'])

        for frame_path in frame_dir.glob('*.jpg'):
            frame = cv2.imread(str(frame_path))
            results = model(frame)
            annotated_frame = results[0].plot()

            object_counts = count_objects(results)
            for obj, count in object_counts.items():
                writer.writerow([frame_path.name, obj, count])

            output_path = output_dir / frame_path.name
            cv2.imwrite(str(output_path), annotated_frame)


def count_objects(results):
    detections = results[0].boxes.data.cpu().numpy()
    object_counts = {}
    for detection in detections:
        class_id = int(detection[-1])
        class_name = model.names[class_id]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
    return object_counts


@app.route('/download')
def download_video():
    input_folder = app.config['PROCESSED_FRAME_FOLDER']
    output_video = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
    create_video(input_folder, output_video)
    # return send_file(output_video, as_attachment=True)
    return render_template('download.html', video_path=output_video)


@app.route('/download_video_file')
def download_video_file():
    output_video = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
    return send_file(output_video, as_attachment=True)


def create_video(input_folder, output_video, fps=1):
    # ffmpeg_binary = iio.get_ffmpeg_exe()
    # (
    #     ffmpeg.input(f'{input_folder}/frame_%04d.jpg', framerate=fps).output(output_video, vcodec='libx264', pix_fmt='yuv420p').run(cmd=ffmpeg_binary)
    # )
    try:
        (
            ffmpeg
            .input(f'{input_folder}/frame_%04d.jpg', framerate=fps)
            .output(output_video, vcodec='libx264', pix_fmt='yuv420p')
            .run(overwrite_output=True)
        )
        print(f"Video saved to : {output_video}")
    except ffmpeg.Error as e:
        print("FFMpeg Error: ", e.stderr.decode())


if __name__ == "__main__":
    app.run(debug=True)
