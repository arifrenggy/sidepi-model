import json, random, numpy as np, tensorflow as tf
tf.get_logger().setLevel('ERROR'); tf.random.set_seed(1201); random.seed(1201)
U = 256
KELAS = ['bercak_coklat','blas','bukan_padi','kresek','penggerek','penggulung','sehat','tungro']
j8 = json.load(open('/tmp/j8.json')); H8 = np.load('/tmp/h8.npy')
butir = []
for f, h in zip(j8, H8):
    kelas = f.split('/')[3]
    butir.append((f, KELAS.index(kelas), h.tobytes().hex()[:16]))
baru = json.load(open('/tmp/robo_baru.json'))
bulat = set()
for b in baru:
    if b['split'] not in ('train','valid'): continue
    g = 'robo_' + b['berkas'].rsplit('/',1)[1]
    butir.append((b['berkas'], KELAS.index(b['kelas']), g))
uji_robo = [b['berkas'] for b in baru if b['split'] == 'test']
grup = sorted(set(b[2] for b in butir)); random.shuffle(grup)
uji_g = set(grup[:int(len(grup)*0.15)])
latih = [b for b in butir if b[2] not in uji_g]
uji = [b for b in butir if b[2] in uji_g]
print('total', len(butir), '| latih', len(latih), '| uji', len(uji), flush=True)

def muat(p, y):
    im = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
    return tf.image.resize(im,(U,U))/255.0, tf.one_hot(y, len(KELAS))
def acak(x,y):
    x = tf.image.rot90(x, k=tf.random.uniform([],0,4,dtype=tf.int32))
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x,0.28); x = tf.image.random_contrast(x,0.72,1.3)
    x = tf.image.random_saturation(x,0.78,1.25); x = tf.image.random_hue(x,0.03)
    x = tf.image.random_crop(tf.image.resize(x,(300,300)),(U,U,3))
    return tf.clip_by_value(x,0,1), y
def buat(d, latih=True):
    ds = tf.data.Dataset.from_tensor_slices(([b[0] for b in d],[b[1] for b in d])).map(muat, num_parallel_calls=4)
    if latih: ds = ds.shuffle(2500).map(acak, num_parallel_calls=4)
    return ds.batch(32).prefetch(2)
dl, du = buat(latih), buat(uji, False)
h = np.bincount([b[1] for b in latih], minlength=len(KELAS)).astype(float)
bobot = {k: float(min(len(latih)/(len(KELAS)*h[k]), 3.0)) for k in range(len(KELAS))}
print('bobot:', {KELAS[k]: round(v,2) for k,v in bobot.items()}, flush=True)

dasar = tf.keras.applications.MobileNetV3Small(input_shape=(U,U,3), include_top=False, weights='imagenet', pooling='avg')
dasar.trainable = False
for l in dasar.layers:
    if isinstance(l, tf.keras.layers.BatchNormalization): l.trainable = True
m = tf.keras.Input(shape=(U,U,3)); x = tf.keras.layers.Rescaling(255.0)(m)
x = dasar(x, training=False); x = tf.keras.layers.Dropout(0.3)(x)
o = tf.keras.layers.Dense(len(KELAS), activation='softmax')(x)
model = tf.keras.Model(m,o)
model.compile(tf.keras.optimizers.Adam(2e-3), tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08), metrics=['accuracy'])
model.fit(dl, validation_data=du, epochs=5, verbose=1, class_weight=bobot,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
dasar.trainable = True
for l in dasar.layers[:-50]: l.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-4), tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08), metrics=['accuracy'])
model.fit(dl, validation_data=du, epochs=4, verbose=1, class_weight=bobot,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
y = np.array([b[1] for b in uji])
acc=[]
for xb,_ in du:
    acc.append((model.predict(xb,verbose=0)+model.predict(tf.image.flip_left_right(xb),verbose=0))/2)
p = np.concatenate(acc); pred = p.argmax(1)
print(f'=== uji internal: {(pred==y).mean()*100:.1f}% ===', flush=True)
for k,n in enumerate(KELAS):
    mm = y==k
    if mm.sum(): print(f'  {n:14} {(pred[mm]==k).mean()*100:5.1f}% (n={int(mm.sum())})', flush=True)
model.export('/tmp/sm12'); open('/tmp/sm12_ok','w').write('1')
print('model tersimpan', flush=True)
