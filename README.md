# Sign-Language
melakukan pelatihan data gambar menggunakan metode randomforest dari bahasa isyarat menjadi text dan bisa bicara

persiapan:
1. cek versi python dengan mengetik python --version di cmd
2. install python 3.11 karena menggunakan mediapipe yang sudah tidak bisa kalau di atas versi 3.11

3. install package:
- open-cv : pip install opencv-python==4.7.0.68
- mediapipe: pip install mediapipe==0.10.5
- gTTS: pip isntall gTTS
- playsound: pip install playsound==1.2.2

penggunaan:
1. run dulu file collect_imgs.py untuk mempersiapkan data gambar bahasa isyarat
2. setelah data sudah di siapkan run file create_dataset.py untuk mengubah file gambar pada data menjadi data.pickle
3. kalau data.pickle sudah selesai di run, lanjut run file train_classifier.py untuk melatih data menggunakan metode random forest
4. nanti akan ada file model.p dan run file inference_classfier.py untuk menjalankan programnya
