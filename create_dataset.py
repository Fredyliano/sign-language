import os                 #buat ngelola direktori atau folder
import pickle             #buat nyimpen data dalam bentuk biner kyk 0.03301305 0.45640239 0.10294068
import mediapipe as mp    #buat deteksi tangan ntr kyk kerangka gtu
import cv2                #buat menampilkan operasi terkait gambar dan video
import matplotlib.pyplot as plt     #buat nampilin gambar

#=== persiapan mediapipe hand gesture ===#
mp_hands = mp.solutions.hands                   #buat deteksi tangan
mp_drawing = mp.solutions.drawing_utils         #buat gambar kerangka tangan
mp_drawing_styles = mp.solutions.drawing_styles #buat gambar 21 titik kerangka tangan

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)    #buat inisialisasi deteksi tangan dalam mode gambar statis

DATA_DIR = './data'                             #direktori tempat nyimpen data gambar

#=== pembuatan dataset yang bakal dilatih ===#
data = []                                       #list buat nampung data fitur hasil ekstrak
labels = []                                     #list buat nampung label tiap kelas gambar
for dir_ in os.listdir(DATA_DIR):               #ngulang buat tiap subfolder di direktori data, misal subfolder 0,1,2,3 dst
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #[1:]  #ngulang buat tiap gambar di subfolder kelas yg sesuai
        data_aux = []                #list bantu buat nampung data fitur sementara buat tiap gambar

        x_ = []                      #list bantu buat nampung koordinat x tiap titik kerangka tangan
        y_ = []                      #list bantu buat nampung koordinat y tiap titik kerangka tangan

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  #baca gambar pake OpenCV
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            #konversi gambar dari BGR ke RGB karena Mediapipe butuh format RGB

        results = hands.process(img_rgb)    #proses gambar pake Mediapipe buat deteksi tangan dan ekstrak titik kerangka tangan
        if results.multi_hand_landmarks:    #kalo ada lebih dari 1 tangan yang terdeteksi
            for hand_landmarks in results.multi_hand_landmarks:   #ngulang buat tiap tangan yang terdeteksi
                for i in range(len(hand_landmarks.landmark)):     #ngulang buat tiap titik kerangka tangan
                    x = hand_landmarks.landmark[i].x              #ambil koordinat x titik kerangka tangan ke-i
                    y = hand_landmarks.landmark[i].y              #ambil koordinat y titik kerangka tangan ke-i

                    x_.append(x)    #simpan koordinat x ke list bantu
                    y_.append(y)    #simpan koordinat y ke list bantu
                    
                for i in range(len(hand_landmarks.landmark)):   #ngulang lagi buat tiap titik kerangka tangan
                    x = hand_landmarks.landmark[i].x            #ambil koordinat x titik kerangka tangan ke-i
                    y = hand_landmarks.landmark[i].y            #ambil koordinat y titik kerangka tangan ke-i
                    data_aux.append(x - min(x_))                #normalisasi koordinat x dan simpan ke data fitur
                    data_aux.append(y - min(y_))                #normalisasi koordinat y dan simpan ke data fitur

#=== visualisasi mediapipe hand gesture ===#                
                # mp_drawing.draw_landmarks(  
                #     img_rgb,    
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()  
                # )   #gambar titik kerangka tangan di gambar RGB buat visualisasi
        
            data.append(data_aux)   #simpan data fitur gambar ke list data
            labels.append(dir_)     #simpan label kelas gambar ke list labels

#         plt.figure(figsize=(5, 5))    #bikin figure buat nampilin gambar
#         plt.imshow(img_rgb)           #tampilin gambar RGB dengan titik kerangka tangan
# plt.show()                            #tampilin semua gambar yang udh diproses
           
#=== simpan file dataset yang bakal dilatih ===#
f = open('data.pickle', 'wb')                       #buka file data.pickle buat nyimpen data fitur dan label
pickle.dump({'data': data, 'labels': labels}, f)    #simpan data fitur dan label dalam file data.pickle
f.close()                                           #tutup file data.pickle
