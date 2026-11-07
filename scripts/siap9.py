import os, json, random, collections
from PIL import Image
random.seed(31)
KELAS = ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']

def ahash(p):
    im = Image.open(p).convert('L').resize((8,8), Image.LANCZOS)
    px = list(im.getdata()); r = sum(px)/64
    return ''.join('1' if v > r else '0' for v in px)

butir = []
for k in KELAS:
    for f in sorted(os.listdir(f'data9/{k}')):
        butir.append((f'data9/{k}/{f}', KELAS.index(k), 'lapang' if f.startswith('lapang_') else 'lab'))
print('total foto:', len(butir), flush=True)

grup = collections.defaultdict(list)
for i, b in enumerate(butir):
    grup[ahash(b[0])].append(i)
kunci = list(grup); random.shuffle(kunci)
batas = int(len(kunci)*0.85)
lama, baru = set(), set()
for i, h in enumerate(kunci):
    (lama if i < batas else baru).update(grup[h])
latih = [i for i in range(len(butir)) if i in lama]
uji = [i for i in range(len(butir)) if i in baru]
print('latih:', len(latih), '| uji:', len(uji), '| kelompok uji:', len(baru), flush=True)
print('uji lapangan:', sum(1 for i in uji if butir[i][2]=='lapang'), flush=True)
for k, nama in enumerate(KELAS):
    print(f'  {nama}: latih {sum(1 for i in latih if butir[i][1]==k)} | uji {sum(1 for i in uji if butir[i][1]==k)}')
json.dump({'_semua': butir, 'latih': latih, 'uji': uji}, open('/tmp/data9_split.json','w'))
