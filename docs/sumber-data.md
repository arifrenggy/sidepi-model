# Sumber data

## 1. Foto laboratorium dan jarak dekat (data8)

9.228 foto, delapan kelas termasuk `bukan_padi` dari 38 spesies tanaman lain.
Berasal dari beberapa dataset Kaggle penyakit daun padi dan tanaman, antara lain
Rice Leaf Diseases, PlantVillage, dan Rice Leaf Disease Image.

Lisensi: campuran, sebagian CC BY 4.0, sebagian tidak dinyatakan. Untuk
pemakaian resmi, periksa ulang lisensi tiap dataset di halaman Kaggle asalnya.

## 2. Foto lapangan berlabel penyakit

10.407 foto dari Kaggle `paddy-disease-classification`
(anthony2261/paddy-disease-classification di Hugging Face), 10 kelas:

```text
0 bacterial_leaf_blight      5 dead_heart
1 bacterial_leaf_streak      6 downy_mildew
2 bacterial_panicle_blight   7 hispa
3 blast                      8 normal
4 brown_spot                 9 tungro
```

Yang dipakai hanya kelas yang cocok dengan aplikasi: blast 1.738, brown_spot
965, bacterial_leaf_blight 479, dead_heart 1.442, normal 1.764, tungro 1.088.
Sisanya dilewati karena tidak ada kelasnya di aplikasi.

## 3. Roboflow Rice Leaf

utopialab/rice-leaf-wfax3, 5.188 foto, 10.906 kotak berlabel, CC BY 4.0,
ekspor COCO 640 x 640. Kelas aslinya 11 dengan label ganda (Leaf Blast dan
Rice Blast, Brown Spot tiga versi), lalu digabung menjadi 8 kelas bersih.

**Temuan penting:** 63% foto Roboflow adalah foto yang sama dengan data latih
kita, hanya diubah ukuran. Setelah semua duplikat dibuang dengan sidik dhash,
yang benar-benar baru hanya 313 foto.

## 4. Zenodo 15817084

1.913 foto, tiga kelas seimbang (Bacterial leaf blight 638, Brown spot 638,
Leaf Blast 637), CC BY 4.0. Hasil augmentasi dari Rice_Data_3 di Kaggle.
Tidak ada satu pun yang sama dengan data latih kita, jadi dipakai sebagai
bahan uji tetap.

## 5. Dataset yang diperiksa dan tidak dipakai

| Dataset | Alasan |
| :--- | :--- |
| GitHub aldrin233/RiceDiseases-DataSet | 334 dari 334 foto sudah ada di data latih |
| DatasetNinja rice-disease (CC0) | 238 dari 373 foto sudah ada di data latih |
| Mendeley n8jpkpfz5p | belum diunduh, butuh akun; 6.758 foto, 10 kelas, ada Neck Blast dan Narrow Brown Spot |

## Duplikat antar dataset

Diukur dengan dhash 64 bit, dianggap duplikat bila jarak Hamming 12 atau kurang.

```text
Roboflow terhadap data latih   63,4% duplikat
DatasetNinja terhadap latih    63,8% duplikat
GitHub aldrin terhadap latih  100,0% duplikat
Zenodo terhadap latih           0,0% duplikat
```
