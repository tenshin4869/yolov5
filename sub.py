import cv2
import torch
import numpy as np
import simpleaudio as sa
import speech_recognition as sr
import threading

# ==== 物体検知の初期設定 ====
# モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')
model.conf = 0.4
classes = model.names
target_classes = ["chair", "couch", "cell phone", "bowl"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] カメラを開けませんでした。")
    exit()

cv2.namedWindow("検出結果", cv2.WINDOW_NORMAL)
cv2.resizeWindow("検出結果", 800, 600)

# 音声ファイルのロード
wave_obj = sa.WaveObject.from_wave_file("/usr/share/sounds/alsa/Front_Center.wav")
played_classes = set()  # 再生済みのクラスを記録

# ==== 音声認識の初期設定 ====
recognizer = sr.Recognizer()
audio_wave_obj = sa.WaveObject.from_wave_file("/usr/share/sounds/alsa/Rear_Center.wav")  # 音声認識で再生するファイル

# 音声認識を停止せずに動作させる関数
def continuous_speech_recognition():
    while True:
        try:
            with sr.Microphone() as source:
                print("音声認識中です。話してください...")
                recognizer.adjust_for_ambient_noise(source)  # 環境音に合わせて調整
                audio_data = recognizer.listen(source)  # 音声を聞き取る

            # 音声認識の実行
            text = recognizer.recognize_google(audio_data, language='ja-JP')
            print("認識されたテキスト: " + text)

            # 特定の言葉が含まれている場合、音声を再生
            if "注文" in text:
                play_audio(audio_wave_obj)

        except sr.UnknownValueError:
            print("音声が理解できませんでした")
        except sr.RequestError:
            print("Google Speech Recognitionサービスに接続できませんでした")

# 音声ファイルを再生する関数
def play_audio(wave_obj):
    play_obj = wave_obj.play()
    play_obj.wait_done()

# ==== 物体検知を実行する関数 ====
def object_detection():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] フレームを取得できませんでした。")
            break

        original_height, original_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (320, 240))
        resized_height, resized_width = frame_resized.shape[:2]

        results = model(frame_resized)
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        detections = results.xyxy[0]

        for *box, confidence, cls in detections.numpy():
            x1, y1, x2, y2 = box
            label = classes[int(cls)]

            if label in target_classes:
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, f"{label} を検出しました！", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if label not in played_classes:
                    play_audio(wave_obj)
                    played_classes.add(label)

        cv2.imshow("検出結果", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== マルチスレッドで並列実行 ====
if __name__ == "__main__":
    # 音声認識スレッド
    speech_thread = threading.Thread(target=continuous_speech_recognition, daemon=True)
    speech_thread.start()

    # 物体検知スレッド
    object_detection()