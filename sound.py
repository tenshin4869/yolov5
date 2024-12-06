import speech_recognition as sr

# Recognizerオブジェクトを作成
recognizer = sr.Recognizer()

# マイクを使用して音声を取得
with sr.Microphone() as source:
    print("何か話してください...")
    recognizer.adjust_for_ambient_noise(source)  # 環境音を調整
    audio_data = recognizer.listen(source)  # 音声を聞き取る

# 音声認識を実行
try:
    text = recognizer.recognize_google(audio_data, language='ja-JP')  # 日本語を指定
    print("認識されたテキスト: " + text)
except sr.UnknownValueError:
    print("音声が理解できませんでした")
except sr.RequestError:
    print("Google Speech Recognitionサービスに接続できませんでした")