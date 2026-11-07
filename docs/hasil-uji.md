# Hasil uji

Semua angka diukur, bukan perkiraan. Setiap uji memakai foto yang tidak dipakai
saat melatih, dipisah per foto sumber memakai sidik dhash.

## Perbandingan versi

| Ujian | v0.11.0 | v0.12.0 |
| :--- | ---: | ---: |
| Pembagian internal per foto sumber | 96,3% | 98,9% |
| Zenodo test (192 foto) | 91,7% | 95,8% |
| Roboflow test, 31 foto baru | 48,4% | 83,9% |
| Foto lapangan berlabel (240 foto) | 12,1% | 16,7% |

## Zenodo test per kelas, v0.12.0

```text
Blas             62/64   96,9%
Bercak coklat    61/64   95,3%
Kresek           53/64   82,8%
```

## Percobaan yang gagal dan tidak dirilis

| Percobaan | Data | Hasil | Keputusan |
| :--- | ---: | :--- | :--- |
| Latih hanya dengan 7.476 foto lapangan | 7.476 | 73,5% pada foto lapangan | tidak dipakai |
| Potongan bercak, kotak seragam | 15.903 | blas 98%, bercak 40% | tidak dipakai |
| Potongan bercak, skala diseragamkan, rotasi 8 arah | 18.233 | blas 96%, bercak 29% | tidak dipakai |
| Potongan bercak dari kotak berlabel DatasetNinja | 1.830 | blas 59%, bercak 96% | tidak dipakai |
| Model publik AishaKanwal ViT 4 kelas | 240 foto | 0% pada blas dan bercak | tidak dipakai |
| Model publik cvmil DINOv2 8 kelas | 240 foto | lapangan 0 sampai 2% | tidak dipakai |
| Model publik anthony2261 ConvNeXt | - | bobot rusak, 20 keluaran untuk 10 label | tidak dipakai |

## Ciri bentuk lesi, diukur langsung

220 foto blas dan 220 foto bercak lapangan, ciri bentuk per bercak:

```text
Ciri          Blas    Bercak    AUC
Kebulatan     0,484   0,517     0,576
Kepadatan     0,405   0,420     0,581
Rasio sisi    1,600   1,640     0,541
Keruncingan   0,636   0,649     0,550
```

AUC mendekati 0,5 berarti ciri bentuk pada data berlabel internet tidak
memisahkan kedua penyakit. Penyebabnya label per tanaman, bukan per bercak.
