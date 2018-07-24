from __future__ import division

import numpy as np

caffe_root = '../caffe-channelwise-dropout/'

import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
import os
import time
import scipy.misc
import copy as copy
import colorsys
import matplotlib
import scipy.io

from skimage import color
from sklearn.cluster import KMeans
from skimage.feature import hog
from scipy import ndimage
from PIL import Image
from regulariser import tv_norm
from random import shuffle
from pylab import *


# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def recursive_search(res_input, scores_input, mat, xx1, yy1, iter_num):
  r,c = mat.shape # r is equal to c
  res = np.zeros((c,2),dtype='float')
  scores = np.zeros((c,1),dtype='float')
  res2 = np.zeros((c,2),dtype='float') - 1
  scores2 = np.zeros((c,1),dtype='float') - 1

  if c == 0:
    return res_input, scores_input, mat, xx1, yy1

  for i in range(c):
    value = np.min(mat[:,i])
    pos = np.argmin(mat[:,i])
    
    res[i,1] = i
    res[i,0] = pos
    scores[i] = value

  if iter_num == 0:
    res2[0,1] = res[0,1]
    res2[0,0] = res[0,0]
    scores2[0] = scores[0]
    num_sample = 1
  else:
    num_sample = 0

  if iter_num == 0:
    for i in range(0,c):
      if i == res[0,0]:
        continue
    
      pos = np.where(res[:,0] == i)[0]
      if pos.shape[0] == 0:
        continue

      min_value = np.min(scores[pos])
      pos = np.where(scores == min_value)[0]

      res2[num_sample,1] = yy1[int(res[pos,1][0])]
      res2[num_sample,0] = xx1[int(res[pos,0][0])]
      scores2[num_sample] = scores[pos]
      num_sample = num_sample + 1
  else:
    res_set = np.unique(res[:,0])
    for i in range(0,len(res_set)):
      pos = np.where(res[:,0] == res_set[i])[0]
      if pos.shape[0] != 0:

        min_value = np.min(scores[pos])
        pos = np.where(scores == min_value)[0]
        
        res2[num_sample,1] = int(res[pos,1][0])
        res2[num_sample,0] = int(res[pos,0][0])
        scores2[num_sample] = scores[pos]
        num_sample = num_sample + 1

  xx = np.array(range(r))
  yy = np.array(range(c))

  xx = np.setdiff1d(xx, res2[:,0])
  yy = np.setdiff1d(yy, res2[:,1])

  mat = mat[xx,:]
  mat = mat[:,yy]

  xx = xx1[xx]
  yy = yy1[yy]

  res2[0:num_sample,0] = xx1[res2[0:num_sample,0].astype('int').tolist()]  
  res2[0:num_sample,1] = yy1[res2[0:num_sample,1].astype('int').tolist()] 

  res2 = np.concatenate((res_input, res2[0:num_sample,:]),axis=0)
  scores2 = np.concatenate((scores_input, scores2[0:num_sample]),axis=0)

  return recursive_search(res2, scores2, mat, xx, yy, iter_num + 1)


# start to train the model, hyperparameter definition
max_iter = 90001 # maximum number of iterations
display_every = 1 # show losses every so many iterations
snapshot_every = 30000 # snapshot every so many iterations
vis_every = 1000 # save visualization results every so many iterations
snapshot_folder = 'snapshots' # where to save the snapshots (and load from)
branch_num = 10

gpu_id = int(sys.argv[1])

if not os.path.exists(snapshot_folder):
  os.makedirs(snapshot_folder)

if not os.path.exists('./train_output'):
  os.makedirs('./train_output')

#initialize the nets
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

generator = caffe.AdamSolver('./solver_prototxt/solver_generator.prototxt')
perceptron = caffe.AdamSolver('./solver_prototxt/solver_perceptron.prototxt')

#load from pre-trained model
perceptron.net.copy_from('../VGG_ILSVRC_16_layers.caffemodel')

# bilinear interpolation for deconvolutions
interp_layers = [k for k in generator.net.params.keys() if 'up' in k]
interp_surgery(generator.net, interp_layers)

perceptual_loss_weight = 1
tv_loss_weight = 1e-10

start_snapshot = 1

start = time.time()

def get_id(sample_lst, sample_set, num):
    if len(sample_lst) < num:
        shuffle(sample_set)
        sample_lst.extend(sample_set)

    sample_id = sample_lst[0:num]
    sample_lst = sample_lst[num:len(sample_lst)]
    
    return sample_id, sample_lst


def get_data(data_id):
    num = len(data_id)
    data_output = np.zeros((num,3,128,128),dtype='float')
    data_original = np.zeros((num,3,128,128),dtype='float')
    landmarks_output = np.zeros((num,68,128,128),dtype='float')
    landmarks_display = np.zeros((num,3,128,128),dtype='float')

    for i in range(num):
       data = np.array(Image.open('/BS/3d_deep_learning/work/dataset/lfw/crop_face128_imgs/%05d.png'%(data_id[i])))
       if len(data.shape) == 2:
         data = np.repeat(data.reshape(128,128,1),3,axis=2)

       data = data[:,:,::-1]
       data2 = data - [104.008,116.669,122.675]
       data = data.transpose([2, 0, 1])
       data2 = data2.transpose([2, 0, 1])
       data_original[i,:,:,:] = data
       data_output[i,:,:,:] = data2

       landmarks = np.load('/BS/3d_deep_learning/work/dataset/lfw/landmarks128/%05d.npy'%(data_id[i]))
       landmarks[landmarks > 127] = 127
       landmarks[landmarks < 0] = 0

       for l in range(68):
           landmarks_output[i, l, int(np.round(landmarks[l,1])), int(np.round(landmarks[l,0]))] = 1
           landmarks_display[i, :, int(np.round(landmarks[l,1])), int(np.round(landmarks[l,0]))] = 1

    return data_output, data_original, landmarks_output, landmarks_display


def hist_lab(data):
    num = data.shape[0]
    output = np.zeros((num,300),dtype='float')
    for i in range(num):
      lab = color.rgb2lab(data[i,:,::4,::4].transpose([1,2,0])/256)
      hist_l =np.histogram(lab[:,:,0].reshape(32*32),bins=np.arange(101))
      hist_a =np.histogram(lab[:,:,1].reshape(32*32),bins=np.arange(101) - 50)
      hist_b =np.histogram(lab[:,:,2].reshape(32*32),bins=np.arange(101) - 50)

      hist_l = hist_l[0]
      hist_l = 1.0 * hist_l / np.sum(hist_l)
      hist_a = hist_a[0]
      hist_a = 1.0 * hist_a / np.sum(hist_a)
      hist_b = hist_b[0]
      hist_b = 1.0 * hist_b / np.sum(hist_b)

      output[i,0:100] = hist_l
      output[i,100:200] = hist_a
      output[i,200:300] = hist_b

    return output

samples = range(1,1+4038)
shuffle(samples)

for index_, text_ in enumerate(samples):
  samples[index_] = int(text_)

sample_lst = samples

# generate random latent codes
code1 = np.random.uniform(0,1,(branch_num,128,1,1))
code1 = (code1>0.667).astype(int)
code2 = np.random.uniform(0,1,(branch_num,128,1,1))
code2 = (code2>0.667).astype(int)

np.save('%s/code1.npy'%snapshot_folder, code1)
np.save('%s/code2.npy'%snapshot_folder, code2)

for it in range(start_snapshot, max_iter):

  sample_id, sample_lst = get_id(sample_lst, samples, 1)

  # neighbor searching for neighbor enhanced loss function
  sample_nn = scipy.io.loadmat('/BS/3d_deep_learning/work/dataset/lfw/face128_nn/%05d.mat'%sample_id[0])
  sample_nn = sample_nn['nn']

  pos = np.argwhere(sample_nn[1,:] < 300)
  sample_nn = sample_nn[0,pos]
  sample_nn = sample_nn.reshape(len(sample_nn))
  sample_nn = np.append(sample_id[0], sample_nn)
  num_valid_sample = sample_nn.shape[0]  

  face_data, face_original, face_normal, normal_map = get_data(sample_nn)
  if num_valid_sample > 7:
    lab_feature = hist_lab(face_original) 
    kmeans = KMeans(n_clusters=7, random_state=0).fit(lab_feature)

    sample_id_final = np.zeros((7),dtype='int64')
    for ci in range(7):
      tmp = np.argwhere(kmeans.labels_ == ci)
      tmp = tmp.reshape(len(tmp))
      if np.sum((tmp == 0).astype('int')) == 1:
        sample_id_final[ci] = 0
      else:
        sample_id_final[ci] = tmp[0]


    loc = np.argwhere(sample_id_final == 0)
    tmp = sample_id_final[loc]
    sample_id_final[loc] = sample_id_final[0]
    sample_id_final[0] = tmp
    num_valid_sample = 7
    face_data = face_data[sample_id_final,:,:,:]
    face_normal = face_normal[sample_id_final,:,:,:]

  input_data = face_normal

  sz_input_data = input_data.shape
  input_data_sample = copy(input_data)
  input_data = np.repeat(input_data[0:1,:,:,:],branch_num,axis=0)    # the first example store the well aligned training data, others are neighbors


  input_data = face_normal
  
  input_data = np.repeat(input_data[0:1,:,:,:],branch_num,axis=0)

  generator.net.blobs['sto_code1'].data[...] = code1
  generator.net.blobs['sto_code2'].data[...] = code2

  generator.net.blobs['data1'].data[:,:,0:128,0:128] = input_data
  generator.net.blobs['data2'].data[...] = (generator.net.blobs['data1'].data[:,:,0::2,0::2] + generator.net.blobs['data1'].data[:,:,0::2,1::2] + generator.net.blobs['data1'].data[:,:,1::2,0::2] + generator.net.blobs['data1'].data[:,:,1::2,1::2])
  generator.net.blobs['data3'].data[...] = (generator.net.blobs['data2'].data[0:1,:,0::2,0::2] + generator.net.blobs['data2'].data[0:1,:,0::2,1::2] + generator.net.blobs['data2'].data[0:1,:,1::2,0::2] + generator.net.blobs['data2'].data[0:1,:,1::2,1::2])
  generator.net.blobs['data4'].data[...] = (generator.net.blobs['data3'].data[:,:,0::2,0::2] + generator.net.blobs['data3'].data[:,:,0::2,1::2] + generator.net.blobs['data3'].data[:,:,1::2,0::2] + generator.net.blobs['data3'].data[:,:,1::2,1::2])
  generator.net.blobs['data5'].data[...] = (generator.net.blobs['data4'].data[:,:,0::2,0::2] + generator.net.blobs['data4'].data[:,:,0::2,1::2] + generator.net.blobs['data4'].data[:,:,1::2,0::2] + generator.net.blobs['data4'].data[:,:,1::2,1::2])
  generator.net.blobs['data6'].data[...] = (generator.net.blobs['data5'].data[:,:,0::2,0::2] + generator.net.blobs['data5'].data[:,:,0::2,1::2] + generator.net.blobs['data5'].data[:,:,1::2,0::2] + generator.net.blobs['data5'].data[:,:,1::2,1::2])
  generator.net.forward_simple()
  generated_img = generator.net.blobs['generated'].data

  # perceptual regulariser
  perceptron.net.blobs['mask1'].data[...] = 1
  perceptron.net.blobs['mask2'].data[...] = 1
  perceptron.net.blobs['mask3'].data[...] = 1
  perceptron.net.blobs['mask4'].data[...] = 1
  perceptron.net.blobs['mask5'].data[...] = 1
  perceptron.net.blobs['mask6'].data[...] = 1
  perceptual_loss1 = 0
  perceptual_loss2 = 0
  perceptual_loss3 = 0
  perceptual_loss4 = 0
  perceptual_loss5 = 0
  perceptual_loss6 = 0

  pool1_data = np.zeros((branch_num,3,129,129),dtype='float')
  pool1_target = np.zeros((num_valid_sample,3,129,129),dtype='float')
  pool2_data = np.zeros((branch_num,64,129,129),dtype='float')
  pool2_target = np.zeros((num_valid_sample,64,129,129),dtype='float')
  pool3_data = np.zeros((branch_num,128,65,65),dtype='float')
  pool3_target = np.zeros((num_valid_sample,128,65,65),dtype='float')
  pool4_data = np.zeros((branch_num,256,33,33),dtype='float')
  pool4_target = np.zeros((num_valid_sample,256,33,33),dtype='float')
  pool5_data = np.zeros((branch_num,512,17,17),dtype='float')
  pool5_target = np.zeros((num_valid_sample,512,17,17),dtype='float')
  fc6_data = np.zeros((branch_num,512,17,17),dtype='float')
  fc6_target = np.zeros((num_valid_sample,512,17,17),dtype='float')

  if it >= 500:
    for iter_i in range(branch_num):
      perceptron.net.blobs['data'].data[:,:,0:128,0:128] = generated_img[iter_i,:,0:128,0:128]
      perceptron.net.blobs['target_data'].data[:,:,0:128,0:128] = (face_data[min(iter_i,num_valid_sample-1):min(iter_i,num_valid_sample-1)+1,:,:,:])/127
      perceptron.net.forward_simple()
      pool1_data[iter_i,:,:,:] = perceptron.net.blobs['data'].data
      pool2_data[iter_i,:,:,:] = perceptron.net.blobs['conv1_2_data'].data
      pool3_data[iter_i,:,:,:] = perceptron.net.blobs['conv2_2_data'].data
      pool4_data[iter_i,:,:,:] = perceptron.net.blobs['conv3_2_data'].data
      pool5_data[iter_i,:,:,:] = perceptron.net.blobs['conv4_2_data'].data
      fc6_data[iter_i,:,:,:] = perceptron.net.blobs['conv5_2_data'].data
      if iter_i < num_valid_sample:
        pool1_target[iter_i,:,:,:] = perceptron.net.blobs['target_data'].data
        pool2_target[iter_i,:,:,:] = perceptron.net.blobs['conv1_2_target'].data
        pool3_target[iter_i,:,:,:] = perceptron.net.blobs['conv2_2_target'].data
        pool4_target[iter_i,:,:,:] = perceptron.net.blobs['conv3_2_target'].data
        pool5_target[iter_i,:,:,:] = perceptron.net.blobs['conv4_2_target'].data
        fc6_target[iter_i,:,:,:] = perceptron.net.blobs['conv5_2_target'].data

    loss_nn_mat = np.zeros((branch_num,num_valid_sample),dtype='float')
    for iter_i in range(branch_num):
      for iter_j in range(num_valid_sample):
        tmp = 0
        tmp = tmp + np.sum(np.abs(pool1_data[iter_i,:,:,:] - pool1_target[iter_j,:,:,:]))
        tmp = tmp + np.sum(np.abs(pool2_data[iter_i,:,:,:] - pool2_target[iter_j,:,:,:]))
        tmp = tmp + np.sum(np.abs(pool3_data[iter_i,:,:,:] - pool3_target[iter_j,:,:,:]))
        tmp = tmp + np.sum(np.abs(pool4_data[iter_i,:,:,:] - pool4_target[iter_j,:,:,:]))
        tmp = tmp + np.sum(np.abs(pool5_data[iter_i,:,:,:] - pool5_target[iter_j,:,:,:]))

        loss_nn_mat[iter_i,iter_j] = tmp/(2*branch_num)

    match_res = np.zeros((0,2))
    score = np.zeros((0,1))
    match_res,score,_,_,_ = recursive_search(match_res, score, loss_nn_mat, np.array(range(branch_num)), np.array(range(num_valid_sample)), 0)

  # initialize different branches
  if it <= 500:
    match_res = np.zeros((1,2))
    match_res[0,0] = it % branch_num
    score = np.array([0])
  elif it <= 4000:
    match_res = match_res[0:1,:]
    score = np.array([0])

  generator.net.clear_param_diffs()
  generator.net.blobs['generated'].diff[...] = 0

  tv_loss, tv_gradient = tv_norm(generated_img)

  perceptual_loss1 = 0
  perceptual_loss2 = 0
  perceptual_loss3 = 0
  perceptual_loss4 = 0
  perceptual_loss5 = 0
  perceptual_loss6 = 0

  for iter_i in range(len(score)):
    perceptron.net.blobs['mask1'].data[...] = 1
    perceptron.net.blobs['mask2'].data[...] = 1
    perceptron.net.blobs['mask3'].data[...] = 1
    perceptron.net.blobs['mask4'].data[...] = 1
    perceptron.net.blobs['mask5'].data[...] = 1
    perceptron.net.blobs['mask6'].data[...] = 1

    perceptron.net.blobs['data'].data[:,:,0:128,0:128] = generated_img[int(match_res[iter_i, 0]):int(match_res[iter_i, 0]+1),:,0:128,0:128]
    perceptron.net.blobs['target_data'].data[:,:,0:128,0:128] = (face_data[int(match_res[iter_i, 1]):int(match_res[iter_i, 1]+1),:,:,:])/127
    perceptron.net.forward_simple()
    perceptron.net.clear_param_diffs()
    perceptron.net.backward_simple()

    perceptual_loss1 = np.sum(np.copy(perceptron.net.blobs['Perceptualloss1'].data)) + perceptual_loss1
    perceptual_loss2 = np.sum(np.copy(perceptron.net.blobs['Perceptualloss2'].data)) + perceptual_loss2
    perceptual_loss3 = np.sum(np.copy(perceptron.net.blobs['Perceptualloss3'].data)) + perceptual_loss3
    perceptual_loss4 = np.sum(np.copy(perceptron.net.blobs['Perceptualloss4'].data)) + perceptual_loss4
    perceptual_loss5 = np.sum(np.copy(perceptron.net.blobs['Perceptualloss5'].data)) + perceptual_loss5

    gradient = perceptron.net.blobs['data'].diff[:,:,0:128,0:128]

    generator.net.blobs['generated'].diff[int(match_res[iter_i, 0]):int(match_res[iter_i, 0]+1),:,:,:] = gradient + tv_gradient[int(match_res[iter_i, 0]):int(match_res[iter_i, 0]+1),:,:,:]*tv_loss_weight

  perceptual_loss1 = perceptual_loss1 / len(score)
  perceptual_loss2 = perceptual_loss2 / len(score)
  perceptual_loss3 = perceptual_loss3 / len(score)
  perceptual_loss4 = perceptual_loss4 / len(score)
  perceptual_loss5 = perceptual_loss5 / len(score)

  generator.net.backward_simple()
  generator.apply_update()   
  generator.increment_iter()

  #display
  if it % display_every == 0:
    print >> sys.stderr, "[%s] Iteration %d: %f seconds" % (time.strftime("%c"), it, time.time()-start)
  
    print >> sys.stderr, "  perceptual loss 1 for generator: %e * %e = %f" % (perceptual_loss1, perceptual_loss_weight, perceptual_loss1*perceptual_loss_weight)
    print >> sys.stderr, "  perceptual loss 2 for generator: %e * %e = %f" % (perceptual_loss2, perceptual_loss_weight, perceptual_loss2*perceptual_loss_weight)
    print >> sys.stderr, "  perceptual loss 3 for generator: %e * %e = %f" % (perceptual_loss3, perceptual_loss_weight, perceptual_loss3*perceptual_loss_weight)
    print >> sys.stderr, "  perceptual loss 4 for generator: %e * %e = %f" % (perceptual_loss4, perceptual_loss_weight, perceptual_loss4*perceptual_loss_weight)
    print >> sys.stderr, "  perceptual loss 5 for generator: %e * %e = %f" % (perceptual_loss5, perceptual_loss_weight, perceptual_loss5*perceptual_loss_weight)
    print >> sys.stderr, "  total variations loss for generator: %e * %e = %f" % (tv_loss, tv_loss_weight, tv_loss*tv_loss_weight)

    start = time.time()

  example = 0

  for iter_i in range(branch_num):
    example_generated_img = generated_img[iter_i]
    if iter_i == 0:
      visualization_results = example_generated_img.transpose([1, 2, 0])*127+[104.008,116.669,122.675]
    else:
      visualization_results = np.concatenate((visualization_results,example_generated_img.transpose([1, 2, 0])*127+[104.008,116.669,122.675]),axis=1)
  
  visualization_results = visualization_results[:,:,::-1]
  visualization_results = visualization_results/255
  generation_output = visualization_results
  gt_output = np.zeros(generation_output.shape,dtype='float')
  color_gt_output = np.zeros(generation_output.shape,dtype='float')

  for iter_i in range(len(score)):
    tmp = face_data[int(match_res[iter_i,1])].transpose([1, 2, 0])+[104.008,116.669,122.675]
    tmp = tmp[:,:,::-1]
    tmp = tmp/255
    gt_output[:,128*int(match_res[iter_i,0]):128*(int(match_res[iter_i,0])+1),:] = tmp


    tmp = normal_map[int(match_res[iter_i,1])]
    tmp = tmp.transpose([1, 2, 0])
    tmp = tmp/255
    color_gt_output[:,128*int(match_res[iter_i,0]):128*(int(match_res[iter_i,0])+1),:] = tmp
    if int(match_res[iter_i,1]) == 0:
      color_gt_output[:,128*int(match_res[iter_i,0]):128*(int(match_res[iter_i,0])+1),1] = 0
      color_gt_output[:,128*int(match_res[iter_i,0]):128*(int(match_res[iter_i,0])+1),2] = 0

  visualization = np.concatenate((gt_output, color_gt_output, generation_output),axis=0)


  if it % vis_every == 0:
    scipy.misc.imsave('./train_output/%05d.png' % (it), visualization)

  #snapshot
  if it % snapshot_every == 0:
    curr_snapshot_folder = snapshot_folder +'/' + str(it)
    print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
    if not os.path.exists(curr_snapshot_folder):
      os.makedirs(curr_snapshot_folder)
    
    generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
    generator.net.save(generator_caffemodel)
    
