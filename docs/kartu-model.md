# Kartu model

## Gerbang tanaman

```text
Arsitektur     MobileNetV3-Large, bobot awal ImageNet
Masukan        224 x 224, skala -1 sampai 1
Keluaran       2 kelas: padi, bukan padi
Ukuran         6,0 MB, float16
Ambang         0,50
Tugas          menolak foto yang bukan daun padi sebelum penyakit dinilai
```

Diuji pada 6.981 foto yang tidak dipakai melatih: padi lapangan 100% diterima,
padi jarak dekat 96,2 sampai 100%, 10 spesies tanaman belum dikenal 100%
ditolak, gulma sawah 98,3% ditolak, daun teh 100% ditolak.

## Model penyakit

```text
Arsitektur     MobileNetV3-Small, bobot awal ImageNet
Masukan        256 x 256, skala 0 sampai 1
Keluaran       8 kelas penyakit dan kondisi
Ukuran         1,86 MB, float16
Pelatihan      dua tahap: kepala 5 epoch, penyempurnaan 4 epoch
Augmentasi     rotasi 90 derajat, cermin, kecerahan, kontras, saturasi, potong acak
Bobot kelas    dihitung terbalik dari jumlah foto, dibatasi maksimum 3
Pembagian      per foto sumber memakai sidik dhash, 15% untuk uji
```

Akurasi internal v0.12.0: 98,9%. Per kelas: bercak coklat 98,2%, blas 96,8%,
bukan padi 99,2%, kresek 100%, penggerek 99,4%, penggulung 100%, sehat 100%,
tungro 100%.

## Keterbatasan

1. Foto lapangan berantakan hanya 16,7% tepat. Model menghafal gaya foto
   dataset, bukan memahami penyakit.
2. Blas dan bercak coklat masih tertukar pada foto lapangan. Dua percobaan
   pemisahan pada tingkat potongan bercak gagal dan tidak dirilis.
3. Foto dari internet bebas (Wikimedia Commons, 86 foto berlabel) dijawab 0%
   tepat, karena sudut, jarak, dan latarnya jauh dari data latih.
4. Model tidak mengenal blas leher malai, hispa, downy mildew, bacterial leaf
   streak, dan narrow brown spot.

## Yang dibutuhkan untuk perbaikan

300 sampai 500 foto dari sawah Sumba, satu helai daun memenuhi layar, jarak
15 sampai 20 cm, cahaya siang tidak langsung, ditambah satu potret dekat satu
bercak. Label diperiksa penyuluh atau POPT.
