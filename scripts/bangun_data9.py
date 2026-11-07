import os, io, json, random, shutil
import pyarrow.parquet as pq
from PIL import Image

peta = {3:'blas', 4:'bercak_coklat', 0:'kresek', 5:'penggerek', 8:'sehat', 9:'tungro'}
os.makedirs('data9', exist_ok=True)
for k in ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']:
    os.makedirs(f'data9/{k}', exist_ok=True)
    for f in sorted(os.listdir(f'data8/{k}')):
        if not os.path.exists(f'data9/{k}/{f}'):
            shutil.copy(f'data8/{k}/{f}', f'data9/{k}/{f}')

hitung = {}
for arsip in ['lapang1.parquet','lapang2.parquet']:
    t = pq.read_table(arsip, columns=['image','label'])
    kol = t.column('image'); lab = t.column('label')
    for i in range(t.num_rows):
        y = lab[i].as_py()
        if y not in peta: continue
        d = kol[i].as_py(); data = d.get('bytes') if isinstance(d, dict) else None
        if not data: continue
        try:
            im = Image.open(io.BytesIO(data)).convert('RGB')
        except Exception:
            continue
        im.thumbnail((440, 440))
        kelas = peta[y]
        n = hitung.get(kelas, 0)
        im.save(f'data9/{kelas}/lapang_{n:04d}.jpg', quality=88)
        hitung[kelas] = n + 1
print('tambahan lapangan:', hitung, flush=True)
for k in sorted(os.listdir('data9')):
    print(f'  {k}: {len(os.listdir("data9/"+k))}')
