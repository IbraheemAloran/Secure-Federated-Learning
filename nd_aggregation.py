import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import time
from sklearn.metrics import roc_auc_score


def simple_mean(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp) #predict a client's model update (Eq.3 from paper)
            #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
                        #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

        pred = np.zeros(len(param_list)) #set list of predictions
        pred[:b] = 1 #actual malicious scores of clients
        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy() #calculate distance=oldparams-newparams
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #print("DISTANCE: ", len(distance))
        #print("PRED: ", len(pred))
        #auc1 = roc_auc_score(pred, distance) #calculate auc score from predictions and distance using old Gradients
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy() #calculate distance=predictedparams-newparams
        #auc2 = roc_auc_score(pred, distance) #calculate auc score from predictions and distance using predicted Gradients
        #print("Detection AUC of Old Gradients: %0.4f; Detection AUC of Predicted Gradients: %0.4f" % (auc1, auc2))

        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1, keepdims=1)

    

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * mean_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return mean_nd, distance


# trimmed mean
def trim(old_gradients, param_list, net, lr, b=0, hvp=None):
    '''
    gradients: the list of gradients computed by the worker devices
    net: the global model
    lr: learning rate
    byz: attack
    f: number of compromised worker devices
    b: trim parameter
    '''
    if hvp is not None:
        #print("OLD GRADIENTS: ", old_gradients)
        #print("PARAM LIST: ", param_list)
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc2 = roc_auc_score(pred, distance)
        #print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    # sort
    sorted_array = nd.array(np.sort(nd.concat(*param_list, dim=1).asnumpy(), axis=-1), ctx=mx.gpu(0))
    #sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    # trim
    n = len(param_list)
    m = n - b * 2
    trim_nd = nd.mean(sorted_array[:, b:(b + m)], axis=-1, keepdims=1)

    # update global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size

    return trim_nd, distance


def median(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
            #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
                        #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc2 = roc_auc_score(pred, distance)
        #print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()

        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    if len(param_list) % 2 == 1:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2]
    else:
        median_nd = nd.concat(*param_list, dim=1).sort(axis=-1)[:, len(param_list) // 2: len(param_list) // 2 + 1].mean(axis=-1, keepdims=1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * median_nd[idx:(idx + param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return median_nd, distance


def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def nearest_distance(gradient, c_p):
    sorted_distance = nd.square(c_p - gradient).sum(axis=1).sort(axis=0)
    return sorted_distance[1].asscalar()

def krum(old_gradients, param_list, net, lr, b=0, hvp=None):
    if hvp is not None:
        pred_grad = []
        distance = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
            #distance.append((1 - nd.dot(pred_grad[i].T, param_list[i]) / (
                        #nd.norm(pred_grad[i]) * nd.norm(param_list[i]))).asnumpy().item())

        pred = np.zeros(100)
        pred[:b] = 1
        distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc1 = roc_auc_score(pred, distance)
        distance = nd.norm((nd.concat(*pred_grad, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #auc2 = roc_auc_score(pred, distance)
        #print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        #distance = nd.norm((nd.concat(*old_gradients, dim=1) - nd.concat(*param_list, dim=1)), axis=0).asnumpy()
        #distance = nd.norm(nd.concat(*param_list, dim=1), axis=0).asnumpy()
        # normalize distance
        distance = distance / np.sum(distance)
    else:
        distance = None

    num_params = len(param_list)
    q = b
    if num_params <= 2:
        # if there are too few clients, randomly pick one as Krum aggregation result
        random_idx = np.random.choice(num_params)
        krum_nd = nd.reshape(param_list[random_idx], shape=(-1, 1))
    else:
        if num_params - b - 2 <= 0:
            q = num_params-3
        v = nd.concat(*param_list, dim=1)
        scores = nd.array([score(gradient, v, q) for gradient in param_list])
        min_idx = int(scores.argmin(axis=0).asscalar())
        krum_nd = nd.reshape(param_list[min_idx], shape=(-1, 1))

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * krum_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return krum_nd, distance
