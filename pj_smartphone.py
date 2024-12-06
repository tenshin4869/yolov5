import cv2
import torch
import numpy as np

# モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')  # yolov5s.ptを使用
model.conf = 0.5  # 信頼度の閾値を設定（デフォルトは0.25）

# COCOクラスのラベル
classes = model.names

# カメラの初期化
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] カメラを開けませんでした。デバイスを確認してください。")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] フレームを取得できませんでした。")
        break

    # フレームのリサイズ（オプション）
    frame_resized = cv2.resize(frame, (640, 480))  # 320x320にリサイズ

    # モデルで推論
    results = model(frame_resized)

    # 検出結果の取得
    detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class

    for *box, confidence, cls in detections.numpy():
        x1, y1, x2, y2 = map(int, box)
        label = classes[int(cls)]

        if label == "cell phone":  # スマートフォンのみを検出
            # バウンディングボックスの描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 警告メッセージ
            cv2.putText(frame, "スマホを構わないでください！", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 結果の表示
    cv2.imshow("YOLOv5 Smartphone Detection", frame)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()