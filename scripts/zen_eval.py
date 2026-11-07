import os, glob, numpy as np, collections
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from scipy import ndimage
LABEL=['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
g=Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_gate.tflite',num_threads=2)
p=Interpreter(model_path='/app/conversations/6aa490e1cadd14dbfc0674ed/server/models/sidepi_model.tflite',num_threads=2)
g.allocate_tensors(); p.allocate_tensors()
gi,go=g.get_input_details()[0],g.get_output_details()[0]
pi,po=p.get_input_details()[0],p.get_output_details()[0]
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
def padi(im):
    x=np.asarray(im.resize((224,224),Image.LANCZOS),dtype='float32')/127.5-1
    a=np.zeros((1,224,224,3),dtype='float32'); a[0]=x
    g.set_tensor(gi['index'],a); g.invoke(); r1=g.get_tensor(go['index'])[0].copy()
    a[0]=x[:,::-1,:]; g.set_tensor(gi['index'],a); g.invoke(); r2=g.get_tensor(go['index'])[0]
    return float((r1[1]+r2[1])/2)
def peny(im):
    x=np.asarray(im.resize((256,256),Image.LANCZOS),dtype='float32')/255.0
    a=np.zeros((1,256,256,3),dtype='float32'); a[0]=x
    p.set_tensor(pi['index'],a); p.invoke(); r1=p.get_tensor(po['index'])[0].copy()
    a[0]=x[:,::-1,:]; p.set_tensor(pi['index'],a); p.invoke(); r2=p.get_tensor(po['index'])[0]
    r=(r1+r2)/2
    return {LABEL[i]:float(r[i]) for i in range(len(LABEL))}
peta={'Bacterial leaf blight':'kresek','Brown spot':'bercak_coklat','Leaf Blast':'blas'}
for split in ['test','valid','train']:
    hasil=collections.defaultdict(lambda:[0,0]); tebak=collections.Counter()
    for kelas,target in peta.items():
        for f in glob.glob(f'/tmp/ds/zenodo/Rice Leaf Disease Dataset/{split}/{kelas}/*.PNG'):
            try: im=Image.open(f)
            except Exception: continue
            c=potong(im)
            if padi(c)<0.35:
                hasil[target][1]+=1; tebak['bukan_padi']+=1; continue
            pr=peny(c); k=max(pr,key=pr.get); tebak[k]+=1
            hasil[target][1]+=1
            if k==target: hasil[target][0]+=1
    tot=sum(v[0] for v in hasil.values()); jum=sum(v[1] for v in hasil.values())
    print(f'{split}: {tot}/{jum} = {tot/max(1,jum)*100:.1f}%', {k:f"{v[0]}/{v[1]}" for k,v in hasil.items()}, '| tebakan:', dict(tebak), flush=True)

