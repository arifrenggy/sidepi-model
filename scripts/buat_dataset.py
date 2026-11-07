"""Menyusun dataset latih SIDEPI yang rapi dari data mentah.

Masukan yang dibutuhkan (unduh sendiri, tautan ada di docs/sumber-data.md):
  data8/<kelas>/*.jpg          foto laboratorium dan jarak dekat, 8 kelas
  lapang1.parquet              foto lapangan berlabel (Kaggle paddy-disease-classification)
  lapang2.parquet
  roboflow/                    ekspor COCO dari Roboflow Rice Leaf (CC BY 4.0)

Keluaran:
  dataset/latih/<kelas>/*.jpg  ukuran 384 px, JPEG kualitas 88
  dataset/uji/<kelas>/*.jpg    foto yang tidak dipakai melatih
  dataset/manifest.csv         berkas, kelas, sumber, lisensi, sidik dhash, split
"""
import io, os, csv, json, random, hashlib
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KELAS = ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
PETA_LAPANG = {3:'blas', 4:'bercak_coklat', 0:'kresek', 5:'penggerek', 8:'sehat', 9:'tungro'}
PETA_ROBO = {'Brown Spot':'bercak_coklat','Leaf Blast':'blas',
             'Bacterial Leaf Blight':'kresek','Healthy':'sehat'}
LISENSI = {'data8':'campuran Kaggle, lihat docs/sumber-data.md',
           'lapang':'Kaggle paddy-disease-classification',
           'roboflow':'CC BY 4.0, utopialab/rice-leaf-wfax3'}
random.seed(31)

def simpan(im, tujuan):
    im = im.convert('RGB'); im.thumbnail((384, 384), Image.LANCZOS)
    os.makedirs(os.path.dirname(tujuan), exist_ok=True)
    im.save(tujuan, quality=88)

def dhash(im):
    a = np.asarray(im.convert('L').resize((9, 8), Image.LANCZOS), dtype='int16')
    return ''.join('1' if v else '0' for v in (a[:, 1:] > a[:, :-1]).flatten())

def kumpulkan():
    butir = []
    for k in KELAS:
        for f in sorted(os.listdir(os.path.join(AKAR, 'data8', k))):
            butir.append({'jalur': os.path.join(AKAR, 'data8', k, f), 'kelas': k,
                          'sumber': 'data8', 'lisensi': LISENSI['data8']})
    for arsip in ['lapang1.parquet', 'lapang2.parquet']:
        p = os.path.join(AKAR, arsip)
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=['image', 'label'])
        for i in range(t.num_rows):
            y = t.column('label')[i].as_py()
            if y not in PETA_LAPANG:
                continue
            d = t.column('image')[i].as_py()
            data = d.get('bytes') if isinstance(d, dict) else None
            if not data:
                continue
            butir.append({'bytes': data, 'kelas': PETA_LAPANG[y],
                          'sumber': 'lapang', 'lisensi': LISENSI['lapang']})
    robo = os.path.join(AKAR, 'roboflow')
    if os.path.isdir(robo):
        for split in ['train', 'valid', 'test']:
            folder = os.path.join(robo, split)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith('.jpg'):
                    continue
                for nama, kelas in PETA_ROBO.items():
                    if nama.lower().replace(' ', '') in f.lower().replace(' ', '').replace('_', ''):
                        butir.append({'jalur': os.path.join(folder, f), 'kelas': kelas,
                                      'sumber': 'roboflow', 'lisensi': LISENSI['roboflow']})
                        break
    return butir

def main():
    butir = kumpulkan()
    print('terkumpul', len(butir))
    # buang duplikat persis memakai sidik dhash
    terlihat, bersih = set(), []
    for b in butir:
        try:
            im = Image.open(io.BytesIO(b['bytes'])) if 'bytes' in b else Image.open(b['jalur'])
            im.load()
        except Exception:
            continue
        h = dhash(im)
        b['dhash'] = h
        b['_im'] = im
        if h in terlihat:
            continue
        terlihat.add(h)
        bersih.append(b)
    print('setelah buang duplikat', len(bersih))

    # pembagian per kelompok sidik supaya foto kembar tidak terpisah
    kunci = sorted({b['dhash'][:12] for b in bersih})
    random.shuffle(kunci)
    uji = set(kunci[:int(len(kunci) * 0.15)])
    manifest = []
    n = {'latih': 0, 'uji': 0}
    for b in bersih:
        split = 'uji' if b['dhash'][:12] in uji else 'latih'
        nama = hashlib.md5((b['kelas'] + b['dhash']).encode()).hexdigest()[:12]
        tujuan = os.path.join(AKAR, 'dataset', split, b['kelas'], nama + '.jpg')
        simpan(b['_im'], tujuan)
        n[split] += 1
        manifest.append({'berkas': os.path.relpath(tujuan, AKAR), 'kelas': b['kelas'],
                         'sumber': b['sumber'], 'lisensi': b['lisensi'],
                         'dhash': b['dhash'], 'split': split})
    with open(os.path.join(AKAR, 'dataset', 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['berkas', 'kelas', 'sumber', 'lisensi', 'dhash', 'split'])
        w.writeheader(); w.writerows(manifest)
    print('selesai', n)
    print('manifest: dataset/manifest.csv')

if __name__ == '__main__':
    main()
