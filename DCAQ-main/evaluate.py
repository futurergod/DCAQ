import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from scipy.stats import t
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

use_gpu = torch.cuda.is_available()
# compute acc and mean confidence
def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def evaluate(args, n_runs, ndatas, labels, classes, n_lsamples, n_ways, n_shot ,dataset_name):

    # classification for each task

    acc_list = []
    acc_dis = {}
    print('Start classification for %d tasks...' % (n_runs))
    for i in tqdm(range(n_runs)):
        
        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()

        # Tukey's transform

        beta = 0.7 # 0.7
        support_data = np.power(support_data[:, ], beta)
        query_data = np.power(query_data[:, ], beta)

        _, dim = support_data.shape

        X_aug = support_data
        Y_aug = support_label

        cls_name = args.cls
        
        X_aug = torch.Tensor(X_aug).cuda()
        Y_aug = torch.LongTensor(Y_aug).cuda()
        query_data = torch.Tensor(query_data).cuda()
        query_label = torch.LongTensor(query_label).cuda()
        
        if cls_name == 'tcpr':
            dims=X_aug.shape[-1]
            approximation=torch.cat((X_aug,query_data),dim=0).mean(dim=0)
            approximation=F.normalize(approximation,dim=-1)

            X_aug = F.normalize(X_aug)
            cos_val=torch.mm(X_aug, approximation.view(-1,1))
            X_aug = X_aug - cos_val*approximation

            query_data = F.normalize(query_data)
            cos_val=torch.mm(query_data,approximation.view(-1,1))
            query_data = query_data - cos_val*approximation

            X_aug = F.normalize(X_aug)
            query_data = F.normalize(query_data)

            prototype=X_aug.view(-1,n_ways,dims).mean(dim=0)
            prototype=F.normalize(prototype,dim=-1)
            scores=torch.mm(query_data,prototype.T)

        elif cls_name == 'dcaq':
            
            # channel wise
            c_aug=1.3
            X_aug=torch.where(X_aug!=0,1/(torch.log(1+1/X_aug))**c_aug,0)
            query_data=torch.where(query_data!=0,1/(torch.log(1+1/query_data))**c_aug,0)

            dims=X_aug.shape[-1]
            X_aug=X_aug.view(-1,n_ways,dims)
            for iters in range(args.iters):
                # compute centroids
                approximation=X_aug.mean(dim=(0,1)).unsqueeze(0)
                approximation=F.normalize(approximation,dim=-1)
                appro_query=query_data.mean(dim=0).unsqueeze(0)
                appro_query=F.normalize(appro_query,dim=-1)
                X_aug = F.normalize(X_aug,dim=-1)
                query_data = F.normalize(query_data,dim=-1)

                # remove centroids and alignment
                theta=args.theta
                X_aug = X_aug - approximation+theta*appro_query
                query_data = query_data - appro_query+theta*approximation
                query_data = F.normalize(query_data,dim=-1)
                X_aug = F.normalize(X_aug,dim=-1)

                # compute prototype
                prototype=X_aug.view(-1,n_ways,dims).mean(dim=0)
                prototype=F.normalize(prototype,dim=-1)
                scores=torch.mm(query_data,prototype.T)
                if iters==0:X_aug0=X_aug.view(-1,n_ways,dims)

                # confidence-aware query optimization
                tk=args.tk
                value,indices=torch.topk(scores,tk,dim=0)
                topksamples=query_data[indices.view(-1)].view(tk,-1,dims)
                prob=F.softmax(value,dim=1).view(tk, -1, 1)
                newX=topksamples*prob
                X_aug=torch.cat((X_aug0,newX),dim=0)
                X_aug = F.normalize(X_aug,dim=-1)
            prototype=X_aug.view(-1,n_ways,dims).mean(dim=0)
            prototype=F.normalize(prototype,dim=-1)
            scores=torch.mm(query_data,prototype.T)
            
        else:
            # cosine metric
            dims=X_aug.shape[-1]
            X_aug=F.normalize(X_aug).view(n_ways,-1,dims)
            query_data = F.normalize(query_data,dim=-1)
            prototype=X_aug.view(-1,n_ways,dims).mean(dim=0)
            prototype=F.normalize(prototype,dim=-1)
            scores=torch.mm(query_data,prototype.T)


        # calculate the accuracy
        acc = accuracy(scores, query_label).detach().cpu().numpy()
        acc_list.append(acc)
    
    print('dataset %s,  %d way %d shot, cls is %s'%(dataset_name, n_ways, n_shot, cls_name), ' ACC is: ',  mean_confidence_interval(acc_list))


def main_train(args, select_class=None):

    dataset_name = args.dataset    #'miniImagenet,tieredImagenet,CUB,cifar'
    n_shot = args.n_shot
    n_ways = args.n_ways
    n_queries = 15


    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}

    FSLTask.loadDataSet(args.dataset)


    FSLTask.setRandomStates(cfg)
    ndatas, classes = FSLTask.GenerateRunSet(end=args.n_runs, cfg=cfg, select_class=select_class)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(args.n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(args.n_runs, n_shot + n_queries, n_ways).clone().view(args.n_runs,n_samples)

    evaluate(args, args.n_runs, ndatas, labels, classes, n_lsamples=n_lsamples, n_ways=n_ways, n_shot=n_shot,  dataset_name=dataset_name)

if __name__ == '__main__':
    # data loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cls' , type=str, default='dcaq', help='cosine/dcaq/tcpr')
    parser.add_argument('--n_shot' , type=int, default=1, help='1/5')
    parser.add_argument('--n_ways' , type=int, default=5, help='2/5/10')
    parser.add_argument('--n_runs', type=int, default=10000)
    parser.add_argument('--tsne', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='miniImagenet')
    # hyperparameters
    parser.add_argument('--iters', type=int, default=9)
    parser.add_argument('--tk', type=int, default=12)
    parser.add_argument('--theta', type=int, default=0.3)
    args = parser.parse_args()

    main_train(args)