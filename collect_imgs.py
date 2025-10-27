import os        #buat mengelola direktori dan file
import cv2       #buat menampilkan operasi terkait gambar dan video

#=== buat direktori folder ===#
DATA_DIR = './data'                    #direktori buat menyimpan data gambar
if not os.path.exists(DATA_DIR):       #klo direktori data ga ada, bikin direktori baru
    os.makedirs(DATA_DIR)              #direktori data bkal berisi subfolder 0,1,2,3 dan seterusnya

#=== persiapan bikin class === #
number_of_classes = 5                  #jumlah kelas yang mau dikumpulin datanya
dataset_size = 100                     #jumlah gambar yang mau dikumpulin per kelas

#=== setting kamera ===#
cap = cv2.VideoCapture(0)                   #membuka kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     #atur lebar frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)     #atur tinggi frame 

for j in range(number_of_classes):                      #lakuin pengambilan gambar buat tiap kelas  
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))     #klo subfolder kelas ga ada, bikin subfolder bru

        print('mengumpulkan data untuk kelas {}'.format(j)) #ngasih tau user kelas brp yg lg diambil datanya

#=== pengambilan gambar untuk dataset ===#   
        done = False                            #variabel kontrol buat loop pengambilan gambar
        while True:
            ret, frame = cap.read()             #baca frame dari kamera
            cv2.putText(frame, 'siap? Tekan "m" untuk mulai:', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)   #tulis instruksi ke layar
            cv2.imshow('frame', frame)          #tampilkan frame ke layar
            if cv2.waitKey(25) == ord('m'):     #tunggu tekan tombol 'm' buat mulai ngambil gambar
                break

        counter = 0                             #variabel buat ngitung jumlah gambar yg udh diambil
        while counter < dataset_size:           #loop sampe jumlah gambar yg diambil sesuai target
            ret, frame = cap.read()             #baca frame dari kamera
            cv2.imshow('frame', frame)          #tampilkan frame ke layar
            cv2.waitKey(25)                     #tunggu bentar biar frame ke-render di layar cma milidetik kok
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) #simpen frame sbg gambar jpg di subfolder kelas yg sesuai

            counter += 1                        #tambah 1 ke counter gambar yg diambil


cap.release()                                   #nutup kamera
cv2.destroyAllWindows()                         #nutup semua frame OpenCV yang masih kebuka