import numpy as np
import sounddevice as sd
import librosa
import joblib
import threading
import subprocess
import time

# 学習済みモデルをロード
model = joblib.load('chewing_detection_model.pkl')

# 音声前処理関数
def preprocess_audio(audio_data):
    # 無音データのチェック
    if np.mean(np.abs(audio_data)) < 1e-4:
        raise ValueError("Audio data is silent or too quiet.")
    return librosa.effects.preemphasis(audio_data)

# 特徴量抽出関数
def extract_features(audio_data, sr=16000):
    try:
        audio_data = preprocess_audio(audio_data)

        # 特徴量抽出
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)

        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.mean(delta_mfccs.T, axis=0),
            np.mean(delta2_mfccs.T, axis=0),
            np.mean(spectral_contrast.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(spectral_flatness.T, axis=0)
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# 警告音を鳴らす関数
def alert():
    print("警告！咀嚼音を検知しました！")
    audio_file = '/usr/share/sounds/alsa/Front_Center.wav'
    subprocess.run(['aplay', audio_file])

# リアルタイム検知関数
def real_time_detection():
    samplerate = 16000
    block_duration = 2.0
    overlap_duration = 1.0
    block_size = int(samplerate * block_duration)
    overlap_size = int(samplerate * overlap_duration)

    audio_buffer = np.zeros(block_size)
    detected_flag = False

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, detected_flag
        if status:
            print("Error:", status)

        if detected_flag:
            return

        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata[:, 0]

        # デバッグ: 入力データのエネルギーを表示
        print(f"Audio energy: {np.mean(np.abs(audio_buffer))}")

        if np.mean(np.abs(audio_buffer)) < 1e-4:
            print("Skipping silent audio block.")
            return

        features = extract_features(audio_buffer)
        if features is not None:
            print("Extracted features successfully.")
            probabilities = model.predict_proba([features])[0]
            chewing_score = probabilities[1]
            print(f"Chewing score: {chewing_score}")
            if chewing_score >= 0.6:
                detected_flag = True
                threading.Thread(target=alert).start()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=overlap_size):
        print("リアルタイム検知を開始します。Ctrl+Cで停止します。")
        try:
            while not detected_flag:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("リアルタイム検知を終了します。")

if __name__ == "__main__":
    real_time_detection()