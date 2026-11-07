import os, io, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from scipy import ndimage

LABEL = ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
g = Interpreter(model_path='sidepi_flutter_gate.tflite', num_threads=2) if os.path.exists('sidepi_flutter_gate.tflite') else Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_gate.tflite', num_threads=2)
p = Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_model.tflite', num_threads=2)
g.allocate_tensors(); p.allocate_tensors()
gi, go = g.get_input_details()[0], g.get_output_details()[0]
pi, po = p.get_input_details()[0], p.get_output_details()[0]

def otsu(v):
    h,_ = np.histogram(v, bins=256, range=(0,256)); h=h.astype('float64')
    o=np.cumsum(h); m=np.cumsum(h*np.arange(256)); tot=h.sum()
    with np.errstate(divide='ignore',invalid='ignore'):
        s=(m[-1]*o/tot-m)**2/(o*(tot-o)/tot**2)
    return int(np.nan_to_num(s).argmax())

def potong(im):
    im = im.convert('RGB'); sisi=max(im.size)
    if sisi>480:
        k=480/sisi; im=im.resize((int(im.width*k),int(im.height*k)), Image.LANCZOS)
    a=np.asarray(im,dtype='float32')
    exg=(2*a[:,:,1]-a[:,:,0]-a[:,:,2]+510)/1020*255
    mask=exg>otsu(exg.ravel())
    porsi=mask.mean()
    if porsi<0.02 or porsi>=0.55: return im, porsi
    lab,n=ndimage.label(mask)
    if n==0: return im,porsi
    uk=ndimage.sum(mask,lab,range(1,n+1)); k=int(np.argmax(uk))+1
    ys,xs=np.nonzero(lab==k)
    if len(ys)<100: return im,porsi
    px=int((xs.max()-xs.min()+1)*0.06); py=int((ys.max()-ys.min()+1)*0.06)
    kot=(max(0,xs.min()-px),max(0,ys.min()-py),min(im.width,xs.max()+px),min(im.height,ys.max()+py))
    return im.crop(kot), porsi

def padi(im):
    x=np.asarray(im.resize((224,224),Image.LANCZOS),dtype='float32')/127.5-1
    a=np.zeros((1,224,224,3),dtype='float32'); a[0]=x
    g.set_tensor(gi['index'],a); g.invoke(); r1=g.get_tensor(go['index'])[0]
    a[0]=x[:,::-1,:]; g.set_tensor(gi['index'],a); g.invoke(); r2=g.get_tensor(go['index'])[0]
    return float((r1[1]+r2[1])/2)

def penyakit(im):
    x=np.asarray(im.resize((256,256),Image.LANCZOS),dtype='float32')/255.0
    a=np.zeros((1,256,256,3),dtype='float32'); a[0]=x
    p.set_tensor(pi['index'],a); p.invoke(); r1=p.get_tensor(po['index'])[0]
    a[0]=x[:,::-1,:]; p.set_tensor(pi['index'],a); p.invoke(); r2=p.get_tensor(po['index'])[0]
    r=(r1+r2)/2
    return {LABEL[i]: float(r[i]) for i in range(len(LABEL))}

for kelas in ['blas','bercak_coklat','kresek']:
    f=sorted(os.listdir(f'commons/{kelas}'))
    benar=0; tebak={}; ditolak=0; total=0
    for x in f:
        try: im=Image.open(f'commons/{kelas}/{x}')
        except Exception: continue
        total+=1
        crop,_=potong(im)
        pg=padi(crop)
        if pg<0.35:
            ditolak+=1; tebak['bukan_padi']=tebak.get('bukan_padi',0)+1; continue
        pr=penyakit(crop); k=max(pr,key=pr.get)
        tebak[k]=tebak.get(k,0)+1
        if k==kelas: benar+=1
    print(f'{kelas:14} total {total:2} | tepat {benar:2} = {benar/max(1,total)*100:5.1f}% | ditolak gerbang {ditolak} | tebakan {tebak}', flush=True)
