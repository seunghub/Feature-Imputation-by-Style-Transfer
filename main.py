import argparse
import numpy as np
import torch
from collections import Counter
import pickle

from data_loader import *
from utils import *
from train_test import *
from models import *

def main(args):

    device = torch.device(find_gpu() if torch.cuda.is_available() else "cpu")
    X, y, num_classes = load_data(args.DATA)

    print(f"DATA label statistics: {Counter(y[:,0])}")
    print(f"DATA domain statistics: {Counter(y[:,1])}")

    num_feature = X.shape[1]
    num_domain = args.DATA

    max_epoch = 10000
    batch_size = 4096
    network_lr = 1e-3
    network_norm_rate = 1e-2
    embedding_size = 128

    train_loader, test_loader = make_data_loader(X,y,batch_size)

    #############
    #  PHASE 1  #
    #############
    if args.LOAD == False:
        feature_extractor = FeatureExtractor(num_feature,embedding_size).to(device)
        class_classifier = Classifier(embedding_size,num_classes).to(device)
        domain_classifier = Classifier(embedding_size,num_domain,domain_clf=True).to(device)
        opt = torch.optim.AdamW(list(feature_extractor.parameters())+list(class_classifier.parameters())+list(domain_classifier.parameters()), lr=network_lr,weight_decay=network_norm_rate)
        network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(len(train_loader) * max_epoch))
        print("[Phase1 START]")
        acc, rec, prc = phase1_train_and_test(max_epoch,train_loader,test_loader,feature_extractor,class_classifier,domain_classifier,opt,network_scheduler,device)
        print("[Phase1 END] acc : {%.3f} rec : {%.3f} prc : {%.3f}"%(acc,rec,prc))
    else:
        print("[Phase1 START]")
        with open('model_weight/feature_extractor.pickle','rb') as fe:
            feature_extractor = pickle.load(fe).to(device)
        with open('model_weight/class_classifier.pickle','rb') as cc:
            class_classifier = pickle.load(cc).to(device)
        with open('model_weight/domain_classifier.pickle','rb') as dc:
            domain_classifier = pickle.load(dc).to(device)
        print("[Phase1 END]")

    #############
    #  PHASE 2  #
    #############
    ct_X, ct_y, _ = load_ct_data()
    tau_X, tau_y, _ = load_tau_data()
    fdg_X, fdg_y, _ = load_fdg_data()
    amy_X, amy_y, _ = load_amy_data()

    # Make dataloader for each biomarkers
    ct_train_loader, ct_test_loader = make_data_loader(ct_X, ct_y, batch_size)
    tau_train_loader, tau_test_loader = make_data_loader(tau_X, tau_y, batch_size)
    fdg_train_loader, fdg_test_loader = make_data_loader(fdg_X, fdg_y, batch_size)
    amy_train_loader, amy_test_loader = make_data_loader(amy_X, amy_y, batch_size)
    
    # Freeze Phase1 elements
    for param in feature_extractor.parameters():
        param.requires_grad = False
    for param in class_classifier.parameters():
        param.requires_grad = False

    network_lr = 1e-4

    generator = Generator(embedding_size, num_feature).to(device)
    discriminator = Discriminator(num_feature).to(device)
    opt = torch.optim.AdamW(list(generator.parameters())+list(discriminator.parameters()), lr=network_lr,weight_decay=network_norm_rate)
    network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(len(train_loader) * max_epoch))
    print('[Phase2 CT START]')
    phase2_train_and_test(max_epoch,[train_loader,ct_train_loader],[test_loader,ct_test_loader],feature_extractor,class_classifier,generator,discriminator,opt,network_scheduler,device)
    print('[Phase2 CT END]')

if __name__ == "__main__":
    RANDOM_STATE = 20230115
    torch.manual_seed(RANDOM_STATE)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='Choose how many biomarkes to be used (2,3,4)')
    parser.add_argument('--DATA', default=4, type=int, help='Choose between [2,4]')
    parser.add_argument('--LOAD', action='store_true', default=False)
    args = parser.parse_args()
    main(args)