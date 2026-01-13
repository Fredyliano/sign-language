import cv2                              #buat menampilkan operasi terkait gambar dan video
import mediapipe as mp                  #buat mendeteksi tangan dan landmarknya
import numpy as np                      #buat operasi numerik
import pickle                           #buat load model yang sudah disimpan
import threading                        #buat menjalankan TTS tanpa menghentikan kamera
from gtts import gTTS                   #buat text to speech (bahasa Indonesia)
from playsound import playsound         #buat memutar suara
import os                               #buat cek file cache TTS  
import time                             #buat delay dan pembatas pemutaran suara

#=== Load model ===#
model_dict = pickle.load(open('./modelSVM.p', 'rb'))    #load model yang sudah disimpan dalam file modelSVM.p
model = model_dict['model']                             #ambil model dari dictionary yang di-load

#=== Kamera ===#
cap = cv2.VideoCapture(0)                           #buka kamera default
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)             #set resolusi lebar kamera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)             #set resolusi tinggi kamera

#=== MediaPipe Hands ===#
mp_hands = mp.solutions.hands                       #buat deteksi tangan
mp_drawing = mp.solutions.drawing_utils             #buat gambar kerangka tangan
mp_drawing_styles = mp.solutions.drawing_styles     #buat gambar 21 titik kerangka tangan

hands = mp_hands.Hands(
    static_image_mode=True,                        
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)                                                 #buat inisialisasi deteksi tangan dalam mode video

#=== Label gesture ===#
labels_dict = {0: 'ini', 1: 'contoh', 2: 'terimakasih', 3: 'bahasa', 4: 'isyarat'}     #dictionary buat mapping label numerik ke karakter gesture

#=== Siapkan TTS ===#
TTS_CACHE = {label: f"tts_cache_{label}.mp3" for label in labels_dict.values()}        #Simpan file TTS di cache lokal
for label, file in TTS_CACHE.items():                                                  #Simpan file TTS di cache lokal
    if not os.path.exists(file):                                                       #cek kalo file suara TTS udah ada di cache lokal
        tts = gTTS(text=label, lang='id')                                              #buat inisialisasi TTS dengan teks dan bahasa Indonesia
        tts.save(file)                                                                 #simpan file suara TTS di cache lokal

#=== Variabel kontrol untuk TTS ===#
last_prediction = None                                                                 #mencegah suara diputar terus walau gesture sama
last_speak_time = 0                                                                    #waktu terakhir suara diputar
SPEAK_DELAY = 1                                                                        #detik minimal antar suara

#=== Fungsi pemutaran suara TTS secara terpisah ===#
def speak_async(file_path):                                                       #fungsi buat memutar suara TTS tanpa ngeblokir kamera                   
    def _speak():                                                                 #fungsi bantu buat memutar suara di thread terpisah           
        global last_speak_time                                                    #akses variabel global buat waktu terakhir suara diputar
        if time.time() - last_speak_time < SPEAK_DELAY:                           #klo gerakan terlalu cepat, skip pemutaran suara
            return
        last_speak_time = time.time()                                             #update waktu terakhir suara diputar
        try:
            playsound(file_path)                                                  #putar suara TTS dari file cache
        except Exception as e:                                                    #tangani kalo ada error pas muter suara
            print(f"[TTS Error] {e}")                                             
    threading.Thread(target=_speak, daemon=True).start()                          #jalankan fungsi _speak di thread terpisah

#=== Loop utama ===#
while True:
    ret, frame = cap.read()                                 #baca frame dari kamera
    if not ret:
        continue                                   

    H, W, _ = frame.shape                                    #ambil dimensi frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       #konversi frame dari BGR ke RGB karena Mediapipe butuh format RGB
    results = hands.process(frame_rgb)                       #proses frame pake Mediapipe buat deteksi tangan dan ekstrak titik kerangka tangan

    if results.multi_hand_landmarks:                         #kalo ada tangan yang terdeteksi
        for hand_landmarks in results.multi_hand_landmarks:  #ngulang buat tiap tangan yang terdeteksi
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )                                                           #gambar titik kerangka tangan di frame buat visualisasi

        x_ = [lm.x for lm in hand_landmarks.landmark]                   #list bantu buat nampung koordinat x tiap titik kerangka tangan
        y_ = [lm.y for lm in hand_landmarks.landmark]                   #list bantu buat nampung koordinat y tiap titik kerangka tangan

        data_aux = []                                                   #list bantu buat nampung data fitur sementara buat tiap frame
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[i].x - min(x_))     #normalisasi koordinat x dan simpan ke data fitur
            data_aux.append(hand_landmarks.landmark[i].y - min(y_))     #normalisasi koordinat y dan simpan ke data fitur

        x1 = int(min(x_) * W) - 10                                      #koordinat x1 kotak pembatas tangan
        y1 = int(min(y_) * H) - 10                                      #koordinat y1 kotak pembatas tangan
        x2 = int(max(x_) * W) + 10                                      #koordinat x2 kotak pembatas tangan
        y2 = int(max(y_) * H) + 10                                      #koordinat y2 kotak pembatas tangan

        prediction = model.predict([np.asarray(data_aux)])              #lakukan prediksi gesture tangan pake model yang sudah dilatih
        predicted_character = labels_dict[int(prediction[0])]           #ambil karakter gesture hasil prediksi dari dictionary

        if predicted_character != last_prediction:                      #kalo prediksi gesture beda dari sebelumnya
            print(f"Predicted: {predicted_character}")                  #tampilin hasil prediksi di console
            speak_async(TTS_CACHE[predicted_character])                 #putar suara TTS secara terpisah
            last_prediction = predicted_character                       #update prediksi terakhir

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)        #gambar kotak pembatas tangan
        cv2.putText(frame, predicted_character, (x1, y1 - 10),             
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)   #tulis karakter gesture hasil prediksi di atas kotak pembatas tangan

    cv2.imshow('Hand Gesture', frame)      #tampilin frame hasil deteksi dan prediksi gesture tangan
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()                     #tutup kamera
cv2.destroyAllWindows()           #tutup semua jendela OpenCV
