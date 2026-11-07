import os, glob, json, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from scipy import ndimage
LABEL = ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
g = Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_gate.tflite', num_threads=2)
p = Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_model.tflite', num_threads=2)
g.allocate_tensors(); p.allocate_tensors()
gi, go = g.get_input_details()[0], g.get_output_details()[0]
pi, po = p.get_input_details()[0], p.get_output_details()[0]
peta = {'Rice_Blast':'blas','Brown_Spot':'bercak_coklat','Bacterial_Blight':'kresek'}

def otsu(v):
    h,_=np.histogram(v,bins=256,range=(0,256)); h=h.astype('float64')
    o=np.cumsum(h); m=np.cumsum(h*np.arange(256)); tot=h.sum()
    with np.errstate(divide='ignore',invalid='ignore'):
        s=(m[-1]*o/tot-m)**2/(o*(tot-o)/tot**2)
    return int(np.nan_to_num(s).argmax())

def potong(im):
    im=im.convert('RGB'); sisi=max(im.size)
    if sisi>480:
        k=480/sisi; im=im.resize((int(im.width*k),int(im.height*k)),Image.LANCZOS)
    a=np.asarray(im,dtype='float32')
    exg=(2*a[:,:,1]-a[:,:,0]-a[:,:,2]+510)/1020*255
    mask=exg>otsu(exg.ravel()); porsi=float(mask.mean())
    if porsi<0.02 or porsi>=0.55: return im
    lab,n=ndimage.label(mask)
    if n==0: return im
    uk=ndimage.sum(mask,lab,range(1,n+1)); k=int(np.argmax(uk))+1
    ys,xs=np.nonzero(lab==k)
    if len(ys)<100: return im
    px=int((xs.max()-xs.min()+1)*0.06); py=int((ys.max()-ys.min()+1)*0.06)
    return im.crop((max(0,xs.min()-px),max(0,ys.min()-py),min(im.width,xs.max()+px),min(im.height,ys.max()+py)))

def padi(im):
    x=np.asarray(im.resize((224,224),Image.LANCZOS),dtype='float32')/127.5-1
    a=np.zeros((1,224,224,3),dtype='float32')
    a[0]=x; g.set_tensor(gi['index'],a); g.invoke(); r1=g.get_tensor(go['index'])[0].copy()
    a[0]=x[:,::-1,:]; g.set_tensor(gi['index'],a); g.invoke(); r2=g.get_tensor(go['index'])[0]
    return float((r1[1]+r2[1])/2)

def penyakit(im):
    x=np.asarray(im.resize((256,256),Image.LANCZOS),dtype='float32')/255.0
    a=np.zeros((1,256,256,3),dtype='float32')
    a[0]=x; p.set_tensor(pi['index'],a); p.invoke(); r1=p.get_tensor(po['index'])[0].copy()
    a[0]=x[:,::-1,:]; p.set_tensor(pi['index'],a); p.invoke(); r2=p.get_tensor(po['index'])[0]
    r=(r1+r2)/2
    return {LABEL[i]: float(r[i]) for i in range(len(LABEL))}

hasil = {'blas':[0,0], 'bercak_coklat':[0,0], 'kresek':[0,0]}
tebakan = []
for split in ['train','valid','test']:
    for a in sorted(glob.glob(f'/tmp/ninja/rice-disease/{split}/ann/*.json')):
        d = json.load(open(a))
        nama = d.get('description') or os.path.basename(a).replace('.jpg.json','')
        jp = f'/tmp/ninja/rice-disease/{split}/img/{nama}'
        if not os.path.exists(jp):
            kandidat = glob.glob(f'/tmp/ninja/rice-disease/{split}/img/{nama}*')
            if not kandidat: continue
            jp = kandidat[0]
        # kelas mayoritas kotak
        kk = {}
        for o in d.get('objects', []):
            kk[o['classTitle']] = kk.get(o['classTitle'], 0) + 1
        if not kk: continue
        dominan = max(kk, key=kk.get)
        target = peta[dominan]
        try:
            im = Image.open(jp)
        except Exception:
            continue
        crop = potong(im)
        if padi(crop) < 0.35:
            tebakan.append(('bukan_padi', target)); hasil[target][1] += 1; continue
        pr = penyakit(crop); k = max(pr, key=pr.get)
        tebakan.append((k, target)); hasil[target][1] += 1
        if k == target: hasil[target][0] += 1

print('Akurasi model kita pada dataset Rice Disease (470 foto, belum pernah dilihat)')
for k,(b,t) in hasil.items():
    print(f'  {k:14} tepat {b:3}/{t:3} = {b/max(1,t)*100:5.1f}%')
tot = sum(v[0] for v in hasil.values()); jum = sum(v[1] for v in hasil.values())
print(f'  Rata-rata     {tot}/{jum} = {tot/max(1,jum)*100:.1f}%')
import collections
print('  sebaran tebakan:', dict(collections.Counter([t[0] for t in tebakan])))
print('  kekeliruan blas->bercak:', sum(1 for t in tebakan if t==('bercak_coklat','blas')),
      '| bercak->blas:', sum(1 for t in tebakan if t==('blas','bercak_coklat')))
