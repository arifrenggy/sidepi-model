import os, glob, json, numpy as np
from PIL import Image
def dhash(p):
    try:
        im = Image.open(p).convert('L').resize((9,8), Image.LANCZOS)
        a = np.asarray(im, dtype='int16')
        return np.packbits((a[:,1:] > a[:,:-1]).flatten())
    except Exception:
        return None

def kumpul(pola, batas=None):
    jalur = sorted(glob.glob(pola, recursive=True))
    if batas: jalur = jalur[:batas]
    out = []
    for f in jalur:
        h = dhash(f)
        if h is not None: out.append((f, h))
    return out

data8 = kumpul('/tmp/data8/*/*.jpg')
print('data latih kita:', len(data8), flush=True)
robo = kumpul('/tmp/ds/roboflow/*/*.jpg')
print('roboflow:', len(robo), flush=True)

H8 = np.stack([h for _, h in data8])
Hr = np.stack([h for _, h in robo])
np.save('/tmp/h8.npy', H8); np.save('/tmp/hr.npy', Hr)
json.dump([f for f, _ in data8], open('/tmp/j8.json','w'))
json.dump([f for f, _ in robo], open('/tmp/jr.json','w'))

# jarak hamming minimum tiap foto roboflow ke data latih
mind = np.empty(len(Hr), dtype=np.int16)
for i in range(len(Hr)):
    xor = H8 ^ Hr[i]
    mind[i] = np.bitwise_count(xor).sum(1).min()
np.save('/tmp/mind.npy', mind)
for t in [0, 4, 8, 12, 16]:
    print(f'roboflow dengan jarak <= {t:2}: {(mind <= t).sum():5} ({(mind<=t).mean()*100:5.1f}%)', flush=True)
print('jarak terkecil median:', int(np.median(mind)), flush=True)
