# SIDEPI Model

Model dan dataset untuk identifikasi penyakit daun padi pada aplikasi SIDEPI,
dipakai Dinas Pertanian dan Pangan Sumba Timur.

Versi model terakhir: **v0.12.0**

## Isi repositori

```
models/            berkas TFLite siap dipakai aplikasi
  sidepi_gate.tflite    gerbang tanaman, padi atau bukan, 6,0 MB
  sidepi_model.tflite   8 kelas penyakit, 1,86 MB
data/              dataset siap latih, 256 px, JPEG kualitas 82
  latih/<kelas>/       13.106 foto
  uji/<kelas>/          2.104 foto
  manifest.csv         daftar berkas, kelas, sumber, lisensi, sidik dhash, split
scripts/           seluruh proses dari data mentah sampai model
docs/              keterangan data, kartu model, dan hasil uji
```

## Kelas

`bercak_coklat`, `blas`, `bukan_padi`, `kresek`, `penggerek`, `penggulung`,
`sehat`, `tungro`

## Hasil terukur

| Ujian | v0.11.0 | v0.12.0 |
| :--- | ---: | ---: |
| Pembagian internal per foto sumber | 96,3% | 98,9% |
| Zenodo 15817084 test (192 foto) | 91,7% | 95,8% |
| Roboflow Rice Leaf test (31 foto baru) | 48,4% | 83,9% |
| Foto lapangan berlabel (240 foto) | 12,1% | 16,7% |

Blas pada v0.12.0: 96,8% di pembagian internal, 62 dari 64 di Zenodo test.
Rincian lengkap dan kegagalan yang tercatat ada di `docs/hasil-uji.md`.

## Memakai model

```python
from ai_edge_litert.interpreter import Interpreter
import numpy as np
from PIL import Image

it = Interpreter(model_path='models/sidepi_model.tflite')
it.allocate_tensors()
masuk, keluar = it.get_input_details()[0], it.get_output_details()[0]

gambar = Image.open('daun.jpg').convert('RGB').resize((256, 256))
x = np.asarray(gambar, dtype='float32')[None] / 255.0
it.set_tensor(masuk['index'], x)
it.invoke()
print(it.get_tensor(keluar['index'])[0])
```

Gerbang tanaman memakai skala -1 sampai 1 dan ukuran 224 piksel:

```python
x = np.asarray(gambar.resize((224, 224)), dtype='float32')[None] / 127.5 - 1
```

Setiap prediksi sebaiknya dijalankan dua kali, foto asli dan cermin, lalu
dirata-rata. Selisih hasilnya kecil, tapi menstabilkan foto yang miring.

## Menyusun ulang dari data mentah

```bash
pip install -r requirements.txt
python scripts/buat_dataset.py     # butuh data8/, lapang1.parquet, lapang2.parquet, roboflow/
python scripts/latih.py            # melatih, butuh CPU 4 inti, sekitar 20 menit
python scripts/konversi.py         # ekspor ke TFLite float16
python scripts/uji.py              # mengukur akurasi
```

Tautan unduh data mentah ada di `docs/sumber-data.md`.

## Sumber data dan lisensi

Rincian per sumber, termasuk temuan duplikat antar dataset, ada di
`docs/sumber-data.md`. Sebagian data berlisensi CC BY 4.0 sehingga nama pembuat
wajib dicantumkan bila hasilnya dipublikasikan.

## Batas yang diketahui

Model kuat pada foto daun yang jelas dan memenuhi bingkai. Model masih lemah
pada foto sawah dengan latar berantakan, daun bertumpuk, dan cahaya berbeda.
Pemisahan blas dan bercak coklat pada foto lapangan belum bisa diandalkan,
karena itu aplikasi menampilkan pertanyaan konfirmasi dan dua foto acuan.
