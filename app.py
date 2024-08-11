from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

os.makedirs('./static/uploads', exist_ok=True)

model = YOLO('best (5).pt')
class_list = ['kiri']

# get the html code
@app.route('/')
def index():
    return render_template('forecasting.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No file part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    video_path = os.path.join('./static/uploads/', file.filename)
    file.save(video_path)
    
    return Response(process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    red_line = 100
    blue_line = 400
    text_color = (255, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)
    count = 0
    passed_red_line = False
    passed_blue_line = False
    is_in = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame)

        cv2.line(frame, (8, 100), (927, 100), red_color, 3)
        cv2.line(frame, (8, blue_line), (927, blue_line), blue_color, 3)

        for info in results:
            if len(info.boxes.conf) != 0 and info.boxes.conf[0] > 0.5:  # Adjusted threshold
                conf = info.boxes.conf[0]
                x1, y1 = int(info.boxes.xyxy[0][0]), int(info.boxes.xyxy[0][1])
                x2, y2 = int(info.boxes.xyxy[0][2]), int(info.boxes.xyxy[0][3])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'Kiri detected {conf:.2f}', (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                if cy < (red_line + 20) and cy > (red_line - 20):
                    passed_red_line = True
                    if not passed_blue_line:
                        is_in = False
                if cy < (blue_line + 20) and cy > (blue_line - 20):
                    passed_blue_line = True

                if passed_blue_line and passed_red_line:
                    if is_in:
                        count += 1
                    else:
                        count -= 1
                    passed_red_line, passed_blue_line = False, False
                    is_in = True

        cv2.putText(frame, f'COUNT: {count}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)