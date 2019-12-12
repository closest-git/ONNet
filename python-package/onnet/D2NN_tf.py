'''
    https://github.com/computational-imaging/opticalCNN
    https://github.com/Lyn-Wu/Lyn/blob/master/DNN
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import  numpy as np
from scipy.misc import  imresize
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io, transform

learning_rate = 0.01
#size = 512
size = 28
delta = 0.03
dL = 0.02
batch_size = 64
batch = 10
#mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
mnist = input_data.read_data_sets("E:/ONNet/data/MNIST/raw",one_hot=True)
c = 3e8
Hz = 0.4e12

def fft_test(N = size):
    s = dL * dL / (N * N)
    if False:
        img_raw = tf.io.read_file("E:/ONNet/data/MNIST/test_2.jpg")
        img_raw = tf.image.decode_jpeg(img_raw)
    else:   #tf.io与skimage.io居然不一样，令人难以理解
        img_raw = io.imread("E:/ONNet/data/MNIST/test_2.jpg")
        #print(img_raw)
    img_tensor = tf.squeeze(img_raw)
    with tf.Session() as sess:
        img_tensor = img_tensor.eval()
        print(img_tensor.shape,img_tensor.dtype)
        #print(img_tensor)

    u0 = tf.cast(img_tensor,dtype=tf.complex64)
    print(u0.shape,H_f.shape);
    u1 = tf.fft2d(u0)
    with tf.Session() as sess:
        print(u0.eval())
        print(u1.eval())
    u1 = H_f * u1
    u2 = tf.ifft2d(u1 )
    with tf.Session() as sess:
        print(u1.eval())
        print(u2.eval())

def Init_H(d=delta, N = size, dL = dL, lmb = c/Hz,theta=0.0):
    # Parameter
    df = 1.0 / dL
    k = np.pi * 2.0 / lmb
    D = dL * dL / (N * lmb)
    # phase
    def phase(i, j):
        i -= N // 2
        j -= N // 2
        return ((i * df) * (i * df) + (j * df) * (j * df))


    ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
    # H
    H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
    H_f = np.fft.fftshift(H)
    #print(H_f);    print(H)
    return H,H_f

H,H_f=Init_H()
#fft_test();             input(...)

def _propogation(u0, N = size, dL = dL):
    df = 1.0 / dL
    return tf.ifft2d(H_f*tf.fft2d(u0)*dL*dL/(N*N))*N*N/dL/dL
  
def propogation(u0,d,function=_propogation):
    return tf.map_fn(function,u0)

def make_random(shape):
    return np.random.random(size = shape).astype('float32')


def add_layer_amp(inputs,amp,phase,size,delta):
    return tf.multiply(propogation(inputs,delta),tf.cast(amp,dtype=tf.complex64))
    #return propogation(inputs,delta)*tf.cast(amp,dtype=tf.complex64)
  
def add_layer_phase_out(inputs,amp,phase,size,delta):
    return propogation(inputs,delta,function=_propogation_phase_out)*tf.math.exp(1j*tf.cast(phase,dtype=tf.complex64))

    
def add_layer_phase_in(inputs,amp,phase,size,delta):
    return propogation(inputs,delta,function=_propogation_phase_in)*tf.cast(amp,dtype=tf.complex64)
  
def _change(input_):
    return imresize(input_.reshape(28,28),(size,size),interp="nearest")

def change(input_):
    return np.array(list(map(_change,input_)))

def rang(arr,shape,size=size,base = 512):
    #return arr[shape[0]*size//base:shape[1]*size//base,shape[2]*size//512:shape[3]*size//512]
    x0 = shape[0] * size // base
    y0 = shape[2] * size // base
    delta = (shape[1]-shape[0])* size // base
    return arr[x0:x0+delta,y0:y0+delta]
  
def reduce_mean(tf_):
    return tf.reduce_mean(tf_)
  
def _ten_regions(a):
    return tf.map_fn(reduce_mean,tf.convert_to_tensor([
        rang(a,(120,170,120,170)),
        rang(a,(120,170,240,290)),
        rang(a,(120,170,360,410)),
        rang(a,(220,270,120,170)),
        rang(a,(220,270,200,250)),
        rang(a,(220,270,280,330)),
        rang(a,(220,270,360,410)),
        rang(a,(320,370,120,170)),
        rang(a,(320,370,240,290)),
        rang(a,(320,370,360,410))
    ]))

def ten_regions(logits):
    return tf.map_fn(_ten_regions,tf.abs(logits),dtype=tf.float32)

def download_text(msg,epoch,MIN=1,MAX=7,name=''):
    print("Download {}".format(name))
    if name == 'Phase':
        MIN = 0
        MAX = 2
    for i in range(MIN,MAX):
        print("{} {}:".format(name,i))
        np.savetxt("{}_Time_{}_layer_{}.txt".format(name,epoch+1,i),msg[i-1])
        print("Done")
        
def download_image(msg,epoch,MIN=1,MAX=7,name=''):
    print(f"Plot images-[{MIN}:{MAX}]")
    if name == 'Phase':
        MIN = 0
        MAX = 2
    for i in range(MIN,MAX):
        #print("Image {}:".format(i))
        plt.figure(dpi=650.24)
        plt.axis('off')
        plt.grid('off')
        plt.imshow(msg[i-1])
        plt.savefig("{}_Time_{}_layer_{}.jpg".format(name,epoch+1,i))
        #print("Done")
        
def download_acc(acc,epoch):
    np.savetxt("Acc{}.txt".format(epoch+1),acc)


with tf.device('/cpu:0'):
    data_x = tf.placeholder(tf.float32,shape=(batch_size,size,size))
    data_y = tf.placeholder(tf.float32,shape=(batch_size,10))

    amp=[
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32)
    ]

    phase = [
        tf.constant(np.random.random(size=(size,size)),dtype=tf.float32),
        tf.constant(np.random.random(size=(size,size)),dtype=tf.float32)
    ]

with tf.variable_scope('FullyConnected'):
    layer_1 = add_layer_amp(tf.cast(data_x,dtype=tf.complex64),amp[0],phase[0],size,delta)
    layer_2 = add_layer_amp(layer_1,amp[1],phase[1],size,delta)
    layer_3 = add_layer_amp(layer_2,amp[2],phase[1],size,delta)
    layer_4 = add_layer_amp(layer_3,amp[3],phase[1],size,delta)
    layer_5 = add_layer_amp(layer_4,amp[4],phase[1],size,delta)
    output_layer = add_layer_amp(layer_5,amp[5],phase[1],size,delta)
    output = _propogation(output_layer)

with tf.variable_scope('Loss'):
    logits_abs = tf.square(tf.nn.softmax(ten_regions(tf.abs(output))))
    loss = tf.reduce_sum(tf.square(logits_abs-data_y))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
with tf.variable_scope('Accuracy'):
    pre_correct = tf.equal(tf.argmax(data_y,1),tf.argmax(logits_abs,1))
    accuracy = tf.reduce_mean(tf.cast(pre_correct,tf.float32))

init = tf.global_variables_initializer()
train_epochs = 20
test_epochs = 5
session = tf.Session()
with tf.device('/gpu:0'):
        session.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        #total_batch = 10

        for epoch in tqdm(range(train_epochs)):
            for batch in tqdm(range(total_batch)):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                session.run(train_op,feed_dict={data_x:change(batch_x),data_y:batch_y})

            loss_,acc = session.run([loss,accuracy],feed_dict={data_x:change(batch_x),data_y:batch_y})
            print("epoch :{} loss:{:.4f} acc:{:.4f}".format(epoch+1,loss_,acc)) 

            with tf.device('/cpu:0'):
                msg_amp = np.array(session.run(amp)) 
                download_text(msg_amp,epoch,name='Amp')
                #download_image(msg_amp,epoch,name='Amp')
print("Optimizer finished")