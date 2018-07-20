# import required modules and models
import numpy as np
import time
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc

caffe_root = '../caffe-channelwise-dropout/'

import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
import os

# Set Caffe to GPU mode, load the net in the test phase for inference, and configure input preprocessing.
gpu_id = int(sys.argv[1])
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

net = caffe.Net('./model/test.prototxt',
                './snapshots/110000/generator.caffemodel',
                caffe.TEST)

label_output = np.zeros((1,1,160,160))

fp = open('./split/test.txt','r')
samples = fp.readlines()
fp.close()

for index_, text_ in enumerate(samples):
  samples[index_] = int(text_)

if not os.path.exists('./test_output'):
  os.makedirs('./test_output')

code_new1 = np.random.uniform(0,1,(100-20,128,1,1))
code_new1 = (code_new1>0.5).astype(int)
code_new2 = np.random.uniform(0,1,(100-20,128,1,1))
code_new2 = (code_new2>0.5).astype(int)

code1 = np.load('./code1.npy')
code2 = np.load('./code2.npy')

code1 = np.concatenate((code1,code_new1),axis = 0)
code2 = np.concatenate((code2,code_new2),axis = 0)

for imgs_iter in range(len(samples)):
  normal = scipy.io.loadmat('/BS/3d_deep_learning/work/dataset/normal/%04d.mat'%(samples[imgs_iter]))
  normal = normal['predns']
  normal = normal.transpose([2, 0, 1])

  net.blobs['data1'].data[0,:,0:96,0:96] = normal
  net.blobs['data2'].data[...] = (net.blobs['data1'].data[0:1,:,0::2,0::2] + net.blobs['data1'].data[0:1,:,0::2,1::2] + net.blobs['data1'].data[0:1,:,1::2,0::2] + net.blobs['data1'].data[0:1,:,1::2,1::2])/4
  net.blobs['data3'].data[...] = (net.blobs['data2'].data[:,:,0::2,0::2] + net.blobs['data2'].data[:,:,0::2,1::2] + net.blobs['data2'].data[:,:,1::2,0::2] + net.blobs['data2'].data[:,:,1::2,1::2])/4
  net.blobs['data4'].data[...] = (net.blobs['data3'].data[:,:,0::2,0::2] + net.blobs['data3'].data[:,:,0::2,1::2] + net.blobs['data3'].data[:,:,1::2,0::2] + net.blobs['data3'].data[:,:,1::2,1::2])/4
  net.blobs['data5'].data[...] = (net.blobs['data4'].data[:,:,0::2,0::2] + net.blobs['data4'].data[:,:,0::2,1::2] + net.blobs['data4'].data[:,:,1::2,0::2] + net.blobs['data4'].data[:,:,1::2,1::2])/4
  net.blobs['data6'].data[...] = (net.blobs['data5'].data[:,:,0::2,0::2] + net.blobs['data5'].data[:,:,0::2,1::2] + net.blobs['data5'].data[:,:,1::2,0::2] + net.blobs['data5'].data[:,:,1::2,1::2])/4


  for ii in range(0,100):

      net.blobs['sto_code1'].data[...] = code1[ii,:,:,:]
      net.blobs['sto_code2'].data[...] = code2[ii,:,:,:]
      net.forward_simple()
      generated_img = net.blobs['generated'].data
    
      if not os.path.isdir('./test_output/%04d'%samples[imgs_iter]):
        os.mkdir('./test_output/%04d'%samples[imgs_iter])


      output = net.blobs['generated'].data[0,:,:,:] 
      
      output = output.transpose([1, 2, 0])*127+[104.008,116.669,122.675]
      output = output[:,:,::-1]
      output = output/255
      np.save('./test_output/%04d/%03d.npy' % (samples[imgs_iter],ii), output)

      output[output > 1] = 1
      output[output < 0] = 0
 
      scipy.misc.imsave('./test_output/%04d/%03d.png' % (samples[imgs_iter],ii), output)

  print('saving %d-th output\n'%(imgs_iter+1))

