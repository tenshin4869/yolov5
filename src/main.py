import cv2
import torch
import numpy as np
import simpleaudio as sa  

# モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt')
model.conf = 0.4

classes = model.names


target_classes = ["chair", "couch", "cell phone","bowl"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] カメラを開けませんでした。")
    exit()

cv2.namedWindow("検出結果", cv2.WINDOW_NORMAL)
cv2.resizeWindow("検出結果", 800, 600)  # 必要に応じてサイズを調整

# 音声ファイルのロード
wave_obj = sa.WaveObject.from_wave_file("/usr/share/sounds/alsa/Front_Center.wav")

# 音声再生済みのクラスを記録するセット
played_classes = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] フレームを取得できませんでした。")
        break


    original_height, original_width = frame.shape[:2]

    # フレームをリサイズ（モデルの入力サイズに合わせる）
    frame_resized = cv2.resize(frame, (320, 240))

    # リサイズ後のフレームサイズを取得
    resized_height, resized_width = frame_resized.shape[:2]

    
    results = model(frame_resized)

    
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    # 検出結果の取得
    detections = results.xyxy[0]


    for *box, confidence, cls in detections.numpy():
        x1, y1, x2, y2 = box
        label = classes[int(cls)]

        if label in target_classes:  # 検出するクラスを特定
            # 座標を元のフレームサイズにスケーリング
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 警告メッセージ
            cv2.putText(frame, f"{label} を検出しました！", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 音声の再生（未再生の場合のみ）
            if label not in played_classes:
                play_obj = wave_obj.play()
                played_classes.add(label)  

    # 結果を表示
    cv2.imshow("検出結果", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()