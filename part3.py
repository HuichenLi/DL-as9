import numpy as np
import os
import time

import torch
from torch.autograd import Variable

from helperFunctions import getUCF101
from helperFunctions import loadSequence

import h5py
import cv2

IMAGE_SIZE = 224
NUM_CLASSES = 101
num_of_epochs = 10

data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

##### save predictions directory
prediction_directory = 'UCF-101-predictions-part3/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')

seq_model = torch.load('3d_resnet.model')
seq_model.cuda()
single_model = torch.load('single_frame.model')
single_model.cuda()

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)
seq_model.eval()
single_model.eval()


for i in range(len(test[0])):
    # predict with single frame
    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    data = np.zeros((nFrames,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[j,:,:,:] = frame
    h.close()
    prediction = np.zeros((nFrames,NUM_CLASSES),dtype=np.float32)

    loop_i = list(range(0,nFrames,200))
    loop_i.append(nFrames)

    for j in range(len(loop_i)-1):
        data_batch = data[loop_i[j]:loop_i[j+1]]

        with torch.no_grad():
            x = np.asarray(data_batch,dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            output = single_model(x)

        prediction[loop_i[j]:loop_i[j+1]] = output.cpu().numpy()

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))

    single_prediction = np.sum(np.log(prediction),axis=0)

    # predict with sequence
    data = loadSequence((test[0][index], False))
    if data.size == 0:
        continue

    x = np.expand_dims(np.asarray(data,dtype=np.float32), axis=0)
    x = Variable(torch.FloatTensor(x)).cuda().contiguous()
    y = test[1][index:index+1]
    y = torch.from_numpy(y).cuda()

    with torch.no_grad():
        h = seq_model.conv1(x)
        h = seq_model.bn1(h)
        h = seq_model.relu(h)
        h = seq_model.maxpool(h)

        h = seq_model.layer1(h)
        h = seq_model.layer2(h)
        h = seq_model.layer3(h)
        h = seq_model.layer4[0](h)

        h = seq_model.avgpool(h)

        h = h.view(h.size(0), -1)
        output = seq_model.fc(h)

    prediction = output.cpu().numpy()

    prediction = np.exp(prediction)/np.sum(np.exp(prediction))
    seq_prediction = np.sum(np.log(prediction),axis=0)

    prediction = (single_prediction + seq_prediction)/2

    filename = filename.replace(data_directory+'UCF-101-hdf5/',prediction_directory)
    if not os.path.isfile(filename):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions',data=prediction)

    argsort_pred = np.argsort(-prediction)[0:10]

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d t:%f (%f,%f,%f)'
          % (i,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))


number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])
np.save('part3_confusion_matrix.npy', confusion_matrix)

