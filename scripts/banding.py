import glob, os, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from scipy import ndimage
LABEL=['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
def siapkan(jalur):
    it=Interpreter(model_path=jalur,num_threads=2); it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]
lama = siapkan('/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_model.tflite')
baru = siapkan('/tmp/sidepi_model12.tflite')

def ramal(paket, im):
    it, mi, mo = paket
    x=np.asarray(im.resize((256,256),Image.LANCZOS),dtype='float32')/255.0
    a=np.zeros((1,256,256,3),dtype='float32'); a[0]=x
    it.set_tensor(mi['index'],a); it.invoke(); r1=it.get_tensor(mo['index'])[0].copy()
    a[0]=x[:,::-1,:]; it.set_tensor(mi['index'],a); it.invoke(); r2=it.get_tensor(mo['index'])[0]
    r=(r1+r2)/2
    return LABEL[int(r.argmax())]

def otsu(v):
    h,_=np.histogram(v,bins=256,range=(0,256)); h=h.astype('float64')
    o=np.cumsum(h); m=np.cumsum(h*np.arange(256)); t=h.sum()
    with np.errstate(divide='ignore',invalid='ignore'):
        s=(m[-1]*o/t-m)**2/(o*(t-o)/t**2)
    return int(np.nan_to_num(s).argmax())
def potong(im):
    im=im.convert('RGB'); s0=max(im.size)
    if s0>480:
        k=480/s0; im=im.resize((int(im.width*k),int(im.height*k)),Image.LANCZOS)
    a=np.asarray(im,dtype='float32'); exg=(2*a[:,:,1]-a[:,:,0]-a[:,:,2]+510)/1020*255
    mask=exg>otsu(exg.ravel()); porsi=float(mask.mean())
    if porsi<0.02 or porsi>=0.55: return im
    lab,n=ndimage.label(mask)
    if n==0: return im
    uk=ndimage.sum(mask,lab,range(1,n+1)); k=int(np.argmax(uk))+1
    ys,xs=np.nonzero(lab==k)
    if len(ys)<100: return im
    px=int((xs.max()-xs.min()+1)*0.06); py=int((ys.max()-ys.min()+1)*0.06)
    return im.crop((max(0,xs.min()-px),max(0,ys.min()-py),min(im.width,xs.max()+px),min(im.height,ys.max()+py)))

def uji(daftar, judul):
    benar_l=benar_b=total=0
    for jalur, target in daftar:
        try: im=Image.open(jalur)
        except Exception: continue
        total+=1
        c=potong(im)
        if ramal(lama,c)==target: benar_l+=1
        if ramal(baru,c)==target: benar_b+=1
    print(f'{judul:32} lama {benar_l:3}/{total:3} = {benar_l/max(1,total)*100:5.1f}%   baru {benar_b:3}/{total:3} = {benar_b/max(1,total)*100:5.1f}%', flush=True)

peta={'Bacterial leaf blight':'kresek','Brown spot':'bercak_coklat','Leaf Blast':'blas'}
for split in ['test']:
    d=[]
    for k,t in peta.items():
        for f in glob.glob(f'/tmp/ds/zenodo/Rice Leaf Disease Dataset/{split}/{k}/*.PNG'): d.append((f,t))
    uji(d, f'Zenodo {split}')

import json
baru_test=json.load(open('/tmp/robo_baru.json'))
d=[(b['berkas'], b['kelas']) for b in baru_test if b['split']=='test']
uji(d, 'Roboflow test (baru saja)')

# foto lapangan berlabel dari data9
d=[]
for kelas in ['blas','bercak_coklat','kresek','tungro','sehat','penggerek']:
    for f in sorted(glob.glob(f'/tmp/data9/{kelas}/lapang_*.jpg'))[:40]: d.append((f,kelas))
uji(d, 'Foto lapangan berlabel')
