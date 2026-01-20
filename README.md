# Sign-Language-Fredyliano
melakukan pelatihan data gambar menggunakan metode SVM dari bahasa isyarat menjadi text

persiapan:
1. cek versi python dengan mengetik python --version di cmd
2. install python 3.11 karena menggunakan mediapipe yang sudah tidak bisa digunakan kalau di atas versi 3.11

3. install package:
- open-cv : pip install opencv-python==4.7.0.68
- mediapipe: pip install mediapipe==0.10.5
- gTTS: pip isntall gTTS
- playsound: pip install playsound==1.2.2

penggunaan:
1. run file collect_imgs.py untuk membuat data gambar bahasa isyarat tapi kalau sudah punya tidak usah di run, di sini karena saya pakai data kaggle jadi tidak perlu run collect img
2. jika data sudah di siapkan buka file create_dataset.py dan ubah isi file DATA_DIR sesuai dengan lokasi file dataset gambarnya lalu run file create_dataset.py untuk mengubah file gambar pada data menjadi text koordinat dan di simpan ke file data_baru.pickle
3. kalau data_baru.pickle sudah muncul dan ingin coba lihat isinya buka file read_pickle.py dan run filenya untuk melihat isi dari data pickle tersebut
4. selanjutnya buka file train_classifier.py dan ganti data_dict sesaui dengan pickle yang di buat sebelumnya lalu run file train_classifier.py untuk melatih data menggunakan metode SVM dan modelnya disimpan menjadi file modelSVM.p
5. jika modelSVM.p sudah ada lalu buka file inference_classfier.py dan ganti model_dict sesuai dengan lokasi file model sebelumnya disimpan lalu run file inference_classfier.py untuk melakukan testing bahasa isyaratnya
