# 🐾 Klasifikasi Gambar Hewan dengan CNN, ResNet50, dan EfficientNetB0

## 👥 Anggota Kelompok

👨‍💻 Galang Dwiwana Thabrani 

👩‍💻 Irma Dwiyanti 

👩‍💻 Irma Rohmatillah 

---

## 📚 Deskripsi Proyek

Proyek ini merupakan studi komparatif arsitektur deep learning **Custom CNN**, **ResNet50**, dan **EfficientNetB0** untuk klasifikasi gambar hewan ke dalam tiga kelas: **kucing**, **anjing**, dan **ular**.  
Proyek ini dibuat oleh kelompok sebagai bagian dari tugas akhir Praktikum Pembelajaran Mesin di **Program Studi Teknik Informatika, UIN Sunan Gunung Djati Bandung**.

Metodologi yang digunakan adalah **CRISP-DM** (Cross Industry Standard Process for Data Mining), yang meliputi proses pemahaman data, pemodelan, evaluasi, dan visualisasi performa model.

---

## 🎯 Tujuan

- Membandingkan performa tiga arsitektur CNN dalam klasifikasi gambar hewan
- Menganalisis akurasi, presisi, recall, dan f1-score dari masing-masing model
- Menjelajahi penggunaan transfer learning dibandingkan CNN buatan sendiri

---


## 📁 Dataset

- 📌 Source: [Kaggle - Animal Image Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset)
- 📷 Total: **3000 JPG images** (256x256 pixels)
- 📦 Classes: `cats`, `dogs`, `snakes` (1000 images/class)

Distribusi Data PerKelas :

![image](https://github.com/user-attachments/assets/abbf708e-ea47-41fd-a473-f63c693536e6)

---

## 🧪 Metode: CRISP-DM

### 1️⃣ Business Understanding  
Tujuan utama adalah menentukan model CNN mana yang paling efektif untuk mengklasifikasikan gambar hewan.

### 2️⃣ Data Understanding  
Dataset divisualisasikan dan diperiksa persebaran labelnya. Semua gambar memiliki ukuran awal 256x256 piksel dan terdiri dari 3 saluran warna (RGB).

### 3️⃣ Data Preparation  
- Gambar di-*resize* ke ukuran **224x224**
- Data pelatihan diberi **augmentasi**: rotasi, zoom, flipping horizontal, shifting
- Data validasi & uji hanya dinormalisasi
- Normalisasi dilakukan dengan `rescale=1./255`
- Label diubah menjadi one-hot encoding

### 4️⃣ Modeling  
Tiga model dikembangkan:

- **Custom CNN**: 4 layer konvolusi + pooling, dilanjutkan dense + dropout, output softmax
- **ResNet50**: pretrained ImageNet, `include_top=False`, GlobalAveragePooling + dense
- **EfficientNetB0**: pretrained ImageNet, arsitektur ringan dan efisien

Semua model dikompilasi dengan:

```python
optimizer=Adam()
loss='categorical_crossentropy'
metrics=['accuracy']
```

---

## 📊 Hasil Evaluasi

Berikut adalah hasil evaluasi dari ketiga model berdasarkan data uji:

| Model           | Akurasi | Loss   | Presisi | Recall | F1-Score | Waktu Training (menit) | Jumlah Epoch |
|-----------------|---------|--------|---------|--------|----------|------------------------|---------------|
| **Custom CNN**      | 0.7400  | 0.5619 | 0.74    | 0.74   | 0.74     | 8.88                   | 19            |
| **ResNet50**        | 0.6533  | 0.7901 | 0.65    | 0.65   | 0.65     | 11.78                  | 25            |
| **EfficientNetB0**  | 0.4178  | 1.0982 | 0.42    | 0.42   | 0.31     | 3.60                   | 7             |


Berikut adalah visualisasi dari Performance ketiga algoritma:

### 1. Algoritma Convolutional Neural Network (CNN)
   
![image](https://github.com/user-attachments/assets/415cf291-3363-43bd-b31f-3a3865112ad6)

### Insight:

Performa:

- Accuracy tertinggi mencapai ~77% (training) dan ~75% (validation)
- Performa terbaik di antara ketiga model
- Menunjukkan learning curve yang sehat dengan training dan validation accuracy yang close dan meningkat bersama

Karakteristik:

- Training stabil selama 18 epoch
- Loss curves menunjukkan konvergensi yang baik
- Sedikit fluktuasi di pertengahan training namun kemudian stabil

### 2. Algoritma ResNet50

![image](https://github.com/user-attachments/assets/0819ed4c-aa42-4dd9-8510-ec333c8701e8)

### Insight:

Performa:

- Accuracy tertinggi sekitar 62% (validation) dan 57% (training)
- Menunjukkan tanda-tanda overfitting yang jelas - validation accuracy konsisten lebih tinggi dari training accuracy, yang tidak umum
- Training berjalan selama 25 epoch dengan peningkatan yang steady

Karakteristik:

- Loss menurun secara konsisten untuk kedua set
- Validation loss lebih rendah dari training loss (anomali yang menunjukkan kemungkinan masalah data atau regularization berlebihan)

### 3. Algoritma EfficientNetB0
   
![image](https://github.com/user-attachments/assets/c82df633-0ab6-44a7-a126-3b8aa1c3822d)

### Insight:

Performa:

- Accuracy sangat rendah (~34% untuk semua set)
- Performa terburuk - model seperti tidak belajar sama sekali
- Flat learning curve menunjukkan model gagal mengekstrak fitur yang berguna

Karakteristik:

- Training hanya 6 epoch
- Loss hampir tidak berubah, menunjukkan model stuck
- Kemungkinan masalah: learning rate terlalu tinggi/rendah, arsitektur tidak cocok untuk dataset, atau masalah preprocessing
