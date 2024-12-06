import speech_recognition as sr

# システム上のマイクデバイスをリスト表示
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")