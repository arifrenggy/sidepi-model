import tensorflow as tf, numpy as np
tf.get_logger().setLevel('ERROR')
conv = tf.lite.TFLiteConverter.from_saved_model('/tmp/sm12')
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
open('/tmp/sidepi_model12.tflite','wb').write(tfl)
print('ukuran model baru: %.2f MB' % (len(tfl)/1048576), flush=True)
it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
mi, mo = it.get_input_details()[0], it.get_output_details()[0]
x = np.random.rand(1,256,256,3).astype('float32')
it.set_tensor(mi['index'], x); it.invoke()
y = it.get_tensor(mo['index'])[0]
print('uji jalan, jumlah kelas:', len(y), '| jumlah probabilitas: %.2f' % y.sum(), flush=True)
