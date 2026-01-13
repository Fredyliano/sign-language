import pickle                        #buat nyimpen data dalam bentuk biner kyk 0.03301305 0.45640239 0.10294068
import numpy as np                   #buat ngelola data dalam bentuk array atau numerik
import matplotlib.pyplot as plt      #buat visualisasi gambar
import seaborn as sns                #buat grafik visualisasi gambar
from sklearn.svm import SVC         #buat metode algoritma SVM
  
from sklearn.model_selection import train_test_split   #buat ngebagi data jadi train dan test
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #buat evaluasi performa model

#=== Panggil data ===#
data_dict = pickle.load(open('./data_baru.pickle', 'rb'))  #panggil data yang mau di training

data = np.asarray(data_dict['data'])                  #array fitur koordinat tangan
labels = np.asarray(data_dict['labels'])              #array kelas

# === Bagi dataset ===#
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)   #data test diambil 20%, dan training berarti 80%, data diacak, proporsi tiap kelas di train dan test bakal tetep seimbang

# === Training model ===#
model = SVC(kernel='rbf', probability=True)           #pake metode algoritma SVM
model.fit(x_train, y_train)                           #latih berdasarkan data training

# === evalusai ===#
y_predict = model.predict(x_test)                     #ngehasilin prediksi kelas buat dataset
score = accuracy_score(y_test, y_predict)             #ngitung berapa banyak prediksi yg bnr dibanding total data

print(f'{score * 100:.2f}% sampel diklasifikasikan dengan benar!') #nampilin hasil akurasi model
print("\n=== Classification Report ===")
print(classification_report(y_test, y_predict))                    #nampilin klasifikasinya kyk accuracy dll

# === simpen model ===#
with open('modelSVM.p', 'wb') as f:
    pickle.dump({'model': model}, f)                           #buat nyimpen data yang udah di latih

# === Visualization Confusion Matrix ===#
cm = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Label Asli')
plt.show()
