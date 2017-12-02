#!/bin/python

# Group info:
# agoel5 Anshuman Goel
# kgondha Kaustubh G Gondhalekar
# ndas Neha Das

#Import libraries for simulation
import tensorflow as tf
import numpy as np
import sys
import time
import horovod.tensorflow as hvd

#Imports for visualization
import PIL.Image

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  name = "lake0_py.jpg"
  if hvd.rank() == 1:
    name = "lake1_py.jpg"  
  with open(name,"w") as f:
      PIL.Image.fromarray(a).save(f, "jpeg")

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
sess = tf.InteractiveSession()
#sess = tf.InteractiveSession(config=config) #Use only for capability 3.0 GPU

# Computational Convenience Functions
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
#  5 point stencil #
  five_point = [[0.0, 1.0, 0.0],
                [1.0, -4., 1.0],
                [0.0, 1.0, 0.0]]

#  9 point stencil #
  nine_point = [[0.25, 1.0, 0.25],
                [1.00, -5., 1.00],
                [0.25, 1.0, 0.25]]

#  13 point stencil #
  thirteen_point = [[0.00, 0.00, 0.125, 0.00, 0.00],
                    [0.00, 0.250, 1.00, 0.250, 0.00],
                    [0.125, 1.00, -5.50, 1.00, 0.125],
                    [0.00, 0.250, 1.00, 0.250, 0.00],
                    [0.00, 0.00, 0.125, 0.00, 0.00]]
						   
  laplace_k = make_kernel(thirteen_point)
  return simple_conv(x, laplace_k)

# Define the PDE
if len(sys.argv) != 4:
	print "Usage:", sys.argv[0], "N npebs num_iter"
	sys.exit()
	
N = int(sys.argv[1])
npebs = int(sys.argv[2])
num_iter = int(sys.argv[3])

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init  = np.zeros([N+2, N], dtype=np.float32)
ut_init = np.zeros([N+2, N], dtype=np.float32)

# Some rain drops hit a pond at random points
if hvd.rank() == 0:
  npebs = (npebs+1)/2
else:
  npebs = npebs/2

for n in range(npebs):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))



#create send and receive buffers
send_buf  = np.zeros([2, N], dtype=np.float32)
recv0_buf  = np.zeros([2, N], dtype=np.float32)
recv1_buf  = np.zeros([2, N], dtype=np.float32)

Send_Buffer  = tf.Variable(send_buf,  name='Send_Buffer')
Recv0_Buffer  = tf.Variable(recv0_buf,  name='Recv0_Buffer')
Recv1_Buffer  = tf.Variable(recv1_buf,  name='Recv1_Buffer')

bcast = tf.group(
  tf.assign(Recv1_Buffer, hvd.broadcast(Send_Buffer, 0)),
  tf.assign(Recv0_Buffer, hvd.broadcast(Send_Buffer, 1)))


fill_row = None
if hvd.rank() == 0:
  #fill bottom 2 rows ka values in send_buffer
  fill_row = tf.scatter_update(Send_Buffer, [0,1], Ut[N-2:N, :])
else:
  #fill top 2 rows ka values in send_buffer
  fill_row = tf.scatter_update(Send_Buffer, [0,1], Ut[2:4, :])

update_row = None
if hvd.rank() == 0:
  #fill bottom 2 rows ka values in send_buffer
  update_row = tf.scatter_update(Ut, [N,N+1], Recv0_Buffer[:, :])
else:
  #update top 2 rows ka values in send_buffer
  update_row = tf.scatter_update(Ut, [0,1], Recv1_Buffer[:, :])

# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run num_iter steps of PDE
start = time.time()
for i in range(num_iter):
  # Step simulation
  step.run({eps: 0.06, damping: 0.03})
  #fill rows
  sess.run(fill_row)
  #broadcast
  bcast.run()
  #update rows
  sess.run(update_row)
  


#communicate
bcast = tf.group(
  tf.assign(Recv1_Buffer, hvd.broadcast(Send_Buffer, 0)),  #Rank 0's send_buffer to Rank 1's recv
  tf.assign(Recv0_Buffer, hvd.broadcast(Send_Buffer, 1)))  #Rank 1's send_buffer to Rank 0's recv



end = time.time()
print('Elapsed time: {} seconds'.format(end - start))  
if hvd.rank() == 0:
  DisplayArray(U.eval()[0:N], rng=[-0.1, 0.1])
else:
  DisplayArray(U.eval()[2:], rng=[-0.1, 0.1])  


