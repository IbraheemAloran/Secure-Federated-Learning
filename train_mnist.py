from __future__ import print_function
import mxnet as mx
import nd_aggregation
from mxnet import nd, autograd, gluon
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import random
import argparse
import byzantine
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
import csv
import matplotlib.pyplot as plt

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
np.warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--net", help="net", default='cnn', type=str, choices=['mlr', 'cnn', 'fcnn', 'resnet'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0002, type=float)
    parser.add_argument("--nworkers", help="# workers", default=100, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=500, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=41, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=28, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='partial_trim', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge'])
    parser.add_argument("--aggregation", help="aggregation rule", default='median', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    return parser.parse_args()


def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def params_convert(net):
    tmp = []
    for param in net.collect_params().values():
        tmp.append(param.data().copy())
    params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
    return params


def clip(a, b, c):
    tmp = nd.minimum(nd.maximum(a, b), c)
    return tmp

#clusters the clients and classifies the cluster with the smallest
#malicious average as honest
def detectHonest(score, nobyz, k):
  estimator = KMeans(n_clusters=k)
  estimator.fit(score.reshape(-1, 1))
  label_pred = estimator.labels_
  print("LABELS: ", label_pred)

  honestLabel = 0
  legend = []
  for i in range(k):
    y = np.ones(len(score[label_pred==i]))
    plt.plot(score[label_pred==i], y, 'o')
    if(np.mean(score[label_pred==honestLabel]) > np.mean(score[label_pred==i])):
      honestLabel = i

  plt.xlabel("Suspicious Score")
  plt.legend(["honest", "not honest", "not honest", "not honest", "not honest", "not honest", "not honest",], loc ="upper right")
  
  plt.grid()
  plt.title("Proposed Method")
  plt.show()
  real_label=np.zeros(len(score))
  real_label = real_label - 1
  real_label[nobyz:]= honestLabel
  print("REAL LABELS: ", real_label)
  print("HONEST VALUE: ", honestLabel)
  
  for i in range(len(score)):
    if real_label[i] == honestLabel:
      real_label[i] = 1
    else:
      real_label[i] = 0
    
    if label_pred[i] == honestLabel:
      label_pred[i] = 1
    else:
      label_pred[i] = 0

  print("LABELS: ", label_pred)
  print("REAL LABELS: ", real_label)
  
  print("Honest Results:")
  acc=len(label_pred[label_pred==real_label])/len(score)
  recall=1-np.sum(label_pred[:nobyz])/nobyz
  fpr=1-np.sum(label_pred[nobyz:])/(len(score)-nobyz)
  fnr=np.sum(label_pred[:nobyz])/nobyz
  print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
  print(silhouette_score(score.reshape(-1, 1), label_pred))
  return label_pred




#clusters the clients and classifies the cluster with the largest
#malicious average as malicious
# def detectMalicious(score, nobyz, k):
#   estimator = KMeans(n_clusters=k)
#   estimator.fit(score.reshape(-1, 1))
#   label_pred = estimator.labels_
#   print("LABELS: ", label_pred)

#   honestLabel = 0
#   for i in range(k):
#     y = np.ones(len(score[label_pred==i]))
#     plt.plot(score[label_pred==i], y, 'o')
#     if(np.mean(score[label_pred==honestLabel]) < np.mean(score[label_pred==i])):
#       honestLabel = i

#   plt.show()
#   real_label=np.ones(100)
#   real_label[:nobyz]= 0
#   print("REAL LABELS: ", real_label)
#   print("MALICIOUS VALUE: ", honestLabel)
  
#   for i in range(100):
#     if label_pred[i] == honestLabel:
#       label_pred[i] = 0
#     else:
#       label_pred[i] = 1

#   print("LABELS: ", label_pred)
#   print("REAL LABELS: ", real_label)
  
#   print("Malicious Results:")
#   acc=len(label_pred[label_pred==real_label])/100
#   recall=1-np.sum(label_pred[:nobyz])/nobyz
#   fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
#   fnr=np.sum(label_pred[:nobyz])/nobyz
#   print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
#   print(silhouette_score(score.reshape(-1, 1), label_pred))



def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    y = np.ones(len(score[label_pred==0]))
    z = np.ones(len(score[label_pred==1]))
    print("LABELS: ", label_pred)
    print("SCORES: ", score)
    plt.plot(score[label_pred==0], y, 'o')
    plt.plot(score[label_pred==1], z, 'o')
    plt.grid()
    plt.legend(['Honest','Malicious'], loc ="upper right")
    plt.xlabel("Suspicious Score")
    plt.title("FLDetector")
    plt.show()


    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(len(score))
    real_label[:nobyz]=0
    print("REAL LABELS: ", real_label)
    y = np.ones(len(score[real_label==0]))
    z = np.ones(len(score[real_label==1]))
    plt.plot(score[real_label==1], z, 'o')
    plt.plot(score[real_label==0], y, 'o')
    plt.legend(['Honest','Malicious'], loc ="upper right")
    plt.grid()
    plt.xlabel("Suspicious Score")
    plt.title("True Labels")
    plt.show()
    print("K=2 Results:")
    acc=len(label_pred[label_pred==real_label])/len(score)
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(len(score)-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    print(silhouette_score(score.reshape(-1, 1), label_pred))
    return label_pred, real_label

def detection1(score, nobyz):
    select_k = 0
    nrefs = 10
    ks = range(1, 11)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    print(gapDiff)
    diffs = [j-i for i, j in zip(gapDiff[:-1], gapDiff[1:])]
    #print(diffs)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        plt.plot(range(1,10), gapDiff, "o")
        plt.grid()
        plt.xlabel("Number of Clusters")
        plt.ylabel("Gap Diff")
        plt.title("Gap Scores")
        plt.show()
        print('Attack Detected!')
        return diffs.index(np.max(diffs))+2



def main(args):
    #select cpu or gpu to process
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:
        #set bacth size from args
        batch_size = args.batch_size

        #set dataset from args and size of image and number of classes
        if args.dataset == 'mnist':
            num_inputs = 28 * 28
            num_outputs = 10
            if args.net == 'mlr':
                input_size = (1, 28 * 28)
            else:
                input_size = (1, 1, 28, 28)
        else:
            raise NotImplementedError

        ##############   MODELS   ###################################
        # Multiclass Logistic Regression
        MLR = gluon.nn.Sequential()
        with MLR.name_scope():
            MLR.add(gluon.nn.Dense(num_outputs))

            #################################################

        vae = gluon.nn.Sequential()
        grad_len = 5120
        with vae.name_scope():
            vae.add(gluon.nn.Dense(500, activation="relu"))
            vae.add(gluon.nn.Dense(100, activation="relu"))
            vae.add(gluon.nn.Dense(500, activation="relu"))
            vae.add(gluon.nn.Dense(grad_len))

        #################################################
        # two-layer fully connected nn
        fcnn = gluon.nn.Sequential()
        with fcnn.name_scope():
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(num_outputs))

        #################################################
        # CNN
        cnn = gluon.nn.Sequential()
        with cnn.name_scope():
            cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # The Flatten layer collapses all axis, except the first one, into one axis.
            cnn.add(gluon.nn.Flatten())
            cnn.add(gluon.nn.Dense(512, activation="relu"))
            cnn.add(gluon.nn.Dense(num_outputs))

        ########################################################################################################################
        #Evaluate accuracy of model
        def evaluate_accuracy(data_iterator, net):

            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):
                if args.net == 'mlr':
                    data = data.as_in_context(ctx).reshape((-1, num_inputs))
                    label = label.as_in_context(ctx)
                else:
                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
                acc.update(preds=predictions, labels=label)

            return acc.get()[1]

        ########################################################################################################################
        
        def evaluate_backdoor(data_iterator, net):
            target = 0
            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                data[:, :, 26, 26] = 1
                data[:, :, 26, 24] = 1
                data[:, :, 25, 25] = 1
                data[:, :, 24, 26] = 1
                remaining_idx = list(range(data.shape[0]))
                for example_id in range(data.shape[0]):
                    if label[example_id] != target:
                        label[example_id] = target
                    else:
                        remaining_idx.remove(example_id)

                output = net(data)
                predictions = nd.argmax(output, axis=1)
                predictions = predictions[remaining_idx]
                label = label[remaining_idx]
                acc.update(preds=predictions, labels=label)

            return acc.get()[1]

        def evaluate_edge_backdoor(data, net):
            acc = mx.metric.Accuracy()
            output = net(data)
            label = nd.ones(len(data)).as_in_context(ctx)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
            return acc.get()[1]

        ########################################################################################################################
        # decide attack type
        if args.byz_type == 'partial_trim':
            # partial knowledge trim attack
            byz = byzantine.partial_trim
        elif args.byz_type == 'full_trim':
            # full knowledge trim attack
            byz = byzantine.full_trim
        elif args.byz_type == 'no':
            byz = byzantine.no_byz
        elif args.byz_type == 'gaussian':
            byz = byzantine.gaussian_attack
        elif args.byz_type == 'mean_attack':
            byz = byzantine.mean_attack
        elif args.byz_type == 'full_mean_attack':
            byz = byzantine.full_mean_attack
        elif args.byz_type == 'dir_partial_krum_lambda':
            byz = byzantine.dir_partial_krum_lambda
        elif args.byz_type == 'dir_full_krum_lambda':
            byz = byzantine.dir_full_krum_lambda
        elif args.byz_type == 'backdoor' or 'dba' or 'edge':
            byz = byzantine.scaling_attack
        elif args.byz_type == 'label_flip':
            byz = byzantine.no_byz
        else:
            raise NotImplementedError

            # decide model architecture
        if args.net == 'cnn':
            net = cnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'fcnn':
            net = fcnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'mlr':
            net = MLR
            net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)
        elif args.net == 'resnet':
            #net = resnet_class(block_class, res_layers, res_channels, **kwargs)
            #net.initialize(mx.init.Xavier(magnitude=2.1415926), ctx=ctx)
            kwargs = {'classes': 10, 'thumbnail': True}
            res_layers = [3, 3, 3]
            res_channels = [16, 16, 32, 64]
            model = 'resnet20orig'
            resnet_class = models.ResNetV1
            block_class = models.BasicBlockV1
            net = resnet_class(block_class, res_layers, res_channels, **kwargs)
            net.initialize(mx.init.Xavier(magnitude=3.1415926), ctx=ctx)
        else:
            raise NotImplementedError

        # define loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # set upt parameters
        num_workers = args.nworkers
        lr = args.lr
        epochs = args.nepochs
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        train_acc_list = []
        distance1 = []
        distance2 = []
        auc_list = []

        # generate a string indicating the parameters
        paraString = str(args.dataset) + "+bias " + str(args.bias) + "+net " + str(
            args.net) + "+nepochs " + str(args.nepochs) + "+lr " + str(
            args.lr) + "+batch_size " + str(args.batch_size) + "+nworkers " + str(
            args.nworkers) + "+nbyz " + str(args.nbyz) + "+byz_type " + str(
            args.byz_type) + "+aggregation " + str(args.aggregation) + ".txt"

        # set up seed
        seed = args.seed
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load dataset
        if args.dataset == 'mnist':
            if args.net == 'mlr':
                def transform(data, label):
                    return data.astype(np.float32) / 255, label.astype(np.float32)

                train_data = mx.gluon.data.DataLoader(
                    mx.gluon.data.vision.datasets.MNIST(train=True, transform=transform), 60000, shuffle=True,
                    last_batch='rollover')
                test_data = mx.gluon.data.DataLoader(
                    mx.gluon.data.vision.datasets.MNIST(train=False, transform=transform), 500, shuffle=False,
                    last_batch='rollover')

            elif args.net == 'cnn' or args.net == 'fcnn' or args.net == 'resnet':
                def transform(data, label):
                    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)

                train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                                      60000, shuffle=True, last_batch='rollover')
                test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 500,
                                                     shuffle=False, last_batch='rollover')

            if args.byz_type == 'edge':
                ardis_images = np.loadtxt('./data/ARDIS_test_2828.csv', dtype='float')
                ardis_labels = np.loadtxt('./data/ARDIS_test_labels.csv', dtype='float')
                indices_seven = np.where(ardis_labels[:, 7] == 1)[0]
                images_seven = ardis_images[indices_seven, :] / 255
                test_edge_data = nd.array(images_seven).as_in_context(ctx).reshape(-1, 1, 28, 28)

        else:
            raise NotImplementedError

            # biased assignment
        bias_weight = args.bias
        other_group_size = (1 - bias_weight) / 9.
        worker_per_group = num_workers / 10

        # assign non-IID training data to each worker
        each_worker_data = [[] for _ in range(num_workers)] #OUTPUT: [[], [], [], [], [], [], [], [], [], [], []]
        each_worker_label = [[] for _ in range(num_workers)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                if args.dataset == 'cifar10' and (args.net == 'cnn'):
                    x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
                elif args.dataset == 'mnist' and args.net == 'resnet':
                    x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
                else:
                    x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
                    #x = x.as_in_context(ctx).reshape(-1, num_inputs)
                y = y.as_in_context(ctx)

                # assign a data point to a group
                upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
                lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()

                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

        # concatenate the data for each worker
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        # perform attacks
        if args.byz_type == 'label_flip':
            for i in range(args.nbyz):
                each_worker_label[i] = (each_worker_label[i] + 1) % 9
        if args.byz_type == 'backdoor':
            for i in range(args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0
        if args.byz_type == 'edge':
            ardis_images = np.loadtxt('./data/ARDIS_train_2828.csv', dtype='float')
            ardis_labels = np.loadtxt('./data/ARDIS_train_labels.csv', dtype='float')
            indices_seven = np.where(ardis_labels[:, 7] == 1)[0]
            images_seven = ardis_images[indices_seven, :] / 255
            images_seven = nd.array(images_seven).as_in_context(ctx).reshape(-1, 1, 28, 28)
            label = nd.ones(len(images_seven)).as_in_context(ctx)
            for i in range(args.nbyz):
                each_worker_data[i] = nd.concat(each_worker_data[i][:150], images_seven[:450], dim=0)
                each_worker_label[i] = nd.concat(each_worker_label[i][:150], label[:450], dim=0)

        if args.byz_type == 'dba':
            for i in range(int(args.nbyz / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 4), int(args.nbyz / 2)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 2), int(args.nbyz * 3 / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz * 3 / 4), args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0

        ### begin training
        #set malicious scores
        #malicious_score = []
        #k = -1
        numMals = args.nbyz
        # while (k != 0):
        #   last_weight = []
        #   last_grad = []
        #   old_grad_list = []
        #   del grad_list
        #   grad_list = []
        #   weight_record = []
        #   grad_record = []
        # #   #print(len(each_worker_data[0]))
        #   print(len(each_worker_data))
        #   #print(len(each_worker_label[0]))
        malicious_score = np.zeros((1, len(each_worker_data)))
        for e in range(epochs):
            # for each worker
            for i in range(len(each_worker_data)):
                batch_x = each_worker_data[i][:]
                batch_y = each_worker_label[i][:]
                # sample a batch
                # minibatch = np.random.choice(range(each_worker_data[i].shape[0]), size=batch_size, replace=False)
                with autograd.record():
                  output = net(batch_x)
                  loss = softmax_cross_entropy(output, batch_y)
                # backward
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null']) #store the gradients in the list

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list] #store the parameters from the gradients in the list

            tmp = []
            for param in net.collect_params().values():
                if param.grad_req != 'null':
                    tmp.append(param.data().copy())
            weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
                    #output = net(each_worker_data[i][:]) #train the network on workers' data and store the output
                    #loss = softmax_cross_entropy(output, each_worker_label[i][:]) #compute the loss of the network for each worker's data
                # backward
                #loss.backward() #back propogation?
                #grad_list.append([param.grad().copy() for param in net.collect_params().values()]) #add the parameters of each worker to the list

            #param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list] #reshape parameters

            #tmp = []
            #for param in net.collect_params().values():
            #    tmp.append(param.data().copy())
            #weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)

            # use lbfgs to calculate hessian vector product for every epoch starting from 50
            if e > 50:
                hvp = lbfgs(args, weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # perform attack in the first 50 iterations in every 100 iterations, starting from the 50th iteration
            #if e%100 >= 50:
            param_list = byz(param_list, numMals)

            if args.net == 'resnet':
                if e > 500:
                    lr = args.lr / 5.
                elif e > 750:
                    lr = args.lr / 20.
                else:
                    lr = args.lr

            
        

            #aggregate the global model based on aggregation rule 
            if args.aggregation == 'trim':
                grad, distance = nd_aggregation.trim(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'simple_mean':
                grad, distance = nd_aggregation.simple_mean(old_grad_list, param_list, net, lr, numMals, hvp)
            elif args.aggregation == 'median':
                grad, distance = nd_aggregation.median(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'krum':
                grad, distance = nd_aggregation.krum(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            else:
                raise NotImplementedError
            
            
            # Update malicious distance score
            #if distance is not None and e > 50:
            #   malicious_score.append(distance)
                #print(malicious_score)


            if distance is not None and e > 50:
                malicious_score = np.row_stack((malicious_score, distance))

            if malicious_score.shape[0] >= 11: #Window size is 10 so it assess the malicious scores when there are 11 or more records of malicious scores
                k = detection1(np.sum(malicious_score[-10:], axis=0), numMals)
                if k: #checks if an attack is detected
                    print('Stop at iteration:', e) #stops the process when malicious client is detected
                    fldPred, realPred = detection(np.sum(malicious_score[-10:], axis=0), numMals) #
                    propPred = detectHonest(np.sum(malicious_score[-10:], axis=0), numMals, k)
                    #detectMalicious(np.sum(malicious_score[-10:], axis=0), args.nbyz, k)
                    test_accuracy = evaluate_accuracy(test_data, net)
                    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
                    print("FLD Predcition: ", fldPred)
                    print("Proposed Predcition: ", propPred)
                    print("Real labels: ", realPred)
                    for i in range (len(fldPred)):
                      if fldPred[i] == 0:
                        each_worker_data.pop(i)
                        each_worker_label.pop(i)
                      if fldPred[i] == 0 and realPred[i] == 0:
                        numMals = numMals - 1
                    print(len(each_worker_label))
                    break

            # update weight record and gradient record
            if e > 0:
                weight_record.append(weight - last_weight)
                grad_record.append(grad - last_grad)

            # free memory & reset the list
            if len(weight_record) > 10:
                del weight_record[0]
                del grad_record[0]

            last_weight = weight
            last_grad = grad
            old_grad_list = param_list
            del grad_list
            grad_list = []

            # compute training accuracy every 10 iterations
            if (e + 1) % 10 == 0:
                train_accuracy = evaluate_accuracy(test_data, net)
                if args.byz_type == 'backdoor' or 'dba':
                    backdoor_sr = evaluate_backdoor(test_data, net)
                    print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, train_accuracy, backdoor_sr))
                elif args.byz_type == 'edge':
                    backdoor_sr = evaluate_edge_backdoor(test_edge_data, net)
                    print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, train_accuracy, backdoor_sr))
                else:
                    print("Epoch %02d. Train_acc %0.4f" % (e, train_accuracy))

                train_acc_list.append(train_accuracy)

            # save the training accuracy every 100 iterations
            if (e + 1) % 100 == 0:
                if (args.dataset == 'mnist' and args.net == 'mlr'):
                    if not os.path.exists('out_mnist_mlr/'):
                        os.mkdir('out_mnist_mlr/')
                    np.savetxt('out_mnist_mlr/' + paraString, train_acc_list, fmt='%.4f')
                    #np.savetxt('out_mnist_mlr/distance1.txt', distance1, fmt='%.4f')
                    #np.savetxt('out_mnist_mlr/distance2.txt', distance2, fmt='%.4f')
                    #np.savetxt('out_mnist_mlr/auc.txt', auc_list, fmt='%.4f')
                elif (args.dataset == 'mnist' and args.net == 'cnn'):
                    if not os.path.exists('out_mnist_cnn/'):
                        os.mkdir('out_mnist_cnn/')
                    np.savetxt('out_mnist_cnn/' + paraString, train_acc_list, fmt='%.4f')
                    #np.savetxt('out_mnist_cnn/distance1.txt', distance1, fmt='%.4f')
                    #np.savetxt('out_mnist_cnn/distance2.txt', distance2, fmt='%.4f')
                    #np.savetxt('out_mnist_cnn/auc.txt', auc_list, fmt='%.4f')
                else:
                    raise NotImplementedError

            # compute the final testing accuracy
            if (e + 1) == args.nepochs:
                test_accuracy = evaluate_accuracy(test_data, net)
                print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
                with open('score1.csv', 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(malicious_score)
      # }


if __name__ == "__main__":
    args = parse_args()
    main(args)
