import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from collections import Counter
import pickle

def phase1_train_and_test(max_epoch, train_loader, test_loader, feature_extractor, class_classifier, domain_classifier, opt, scheduler, device):
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, max_epoch + 1):
        feature_extractor.train()
        class_classifier.train()
        domain_classifier.train()

        #Class Accuracy
        train_true_y_list = []
        train_pred_y_list = []

        #Domain Accuracy
        domain_true_y_list = []
        domain_pred_y_list = []

        for batch_idx, (data, target) in enumerate(train_loader):
            p = float(batch_idx + epoch * len(train_loader)) / (max_epoch * len(train_loader))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            data = data.float().to(device)
            target = target.long().to(device)
            opt.zero_grad()

            embedding = feature_extractor(data)
            pred_class = class_classifier(embedding)
            pred_domain = domain_classifier(embedding, alpha)

            class_loss = criterion(pred_class, target[:,0])
            domain_loss = criterion(pred_domain, target[:,1])
            loss = class_loss + domain_loss

            if epoch %200==0:
                pred_y = torch.argmax(pred_class.detach(), axis=1)
                train_pred_y_list.extend(pred_y.tolist())
                train_true_y_list.extend(target[:,0].tolist())

                domain_pred_y = torch.argmax(pred_domain.detach(), axis=1)
                domain_pred_y_list.extend(domain_pred_y.tolist())
                domain_true_y_list.extend(target[:,1].tolist())

            loss.backward()
            opt.step()
            scheduler.step()

        if epoch%1000 ==0:
            print("domain Grount Truth :",Counter(domain_true_y_list))
            print("domain prediction :",Counter(domain_pred_y_list))

        if epoch%200 == 0:
            feature_extractor.eval()
            class_classifier.eval()
            domain_classifier.eval()

            test_true_y_list = []
            test_pred_y_list = []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.float().to(device)
                    target = target.long().to(device)
                    embedding = feature_extractor(data)
                    pred_class = class_classifier(embedding)
                    pred_y = torch.argmax(pred_class.detach(), axis=1)

                    test_pred_y_list.extend(pred_y.tolist())
                    test_true_y_list.extend(target[:,0].tolist())
                
                test_acc = accuracy_score(test_true_y_list, test_pred_y_list)
                test_rec = recall_score(test_true_y_list, test_pred_y_list,average='macro')
                test_prc = precision_score(test_true_y_list, test_pred_y_list,average='macro')
                print("[TEST EPOCH {%d}] acc : {%.3f} rec : {%.3f} prc : {%.3f}"%(epoch,test_acc,test_rec,test_prc))

    with open('model_weight/feature_extractor.pickle','wb') as fe:
        pickle.dump(feature_extractor,fe)
    with open('model_weight/class_classifier.pickle','wb') as cc:
        pickle.dump(class_classifier, cc)
    with open('model_weight/domain_classifier.pickle','wb') as dc:
        pickle.dump(domain_classifier, dc)

    return test_acc, test_rec, test_prc

def phase2_train_and_test(max_epoch, train_loaders, test_loaders, feature_extractor, class_classifier, generator, discriminator, opt, scheduler, device):
    
    # To generate 'fake' biomarkers. Data from all kind of biomarker is included here.
    fake_train_loader = train_loaders[0]
    fake_test_loader = test_loaders[0]
    # 'real' biomarker. Only specific biomarker data is included here.
    real_train_loader = train_loaders[1]
    real_test_loader = test_loaders[1]

    ce_loss = torch.nn.CrossEntropyLoss()

    feature_extractor.eval()
    class_classifier.eval()

    for epoch in range(1, max_epoch + 1):
        generator.train()
        discriminator.train()

        for batch_idx, ((fake_data, fake_target),(real_data,real_target)) in enumerate(zip(fake_train_loader,real_train_loader)):
            
            opt.zero_grad()
            batch_len = np.min([fake_data.shape[0],real_data.shape[0]])
            
            fake_data = fake_data[:batch_len].float().to(device)
            fake_target = fake_target[:batch_len].long().to(device)
            real_data = real_data[:batch_len].float().to(device)
            real_target = real_target[:batch_len].long().to(device)

            embedding = feature_extractor(fake_data)
            fake_generated_data = generator(embedding)

            fake_discr_val = discriminator(fake_generated_data)
            fake_discr_loss = -1 * torch.sum(torch.log(1. - fake_discr_val), axis=0)
            real_discr_val = discriminator(real_data)
            real_discr_loss = -1 * torch.sum(torch.log(real_discr_val), axis=0)
            style_loss = fake_discr_loss + real_discr_loss
            
            pred_class = class_classifier(feature_extractor(fake_generated_data))
            # content_loss = ce_loss(pred_class, fake_target[:,0])
            content_loss = torch.dist(feature_extractor(fake_generated_data),embedding)

            if epoch%200 == 0:
                print("[TRAIN EPOCH {%d}] style loss : {%.3f} content loss : {%.3f}"%(epoch,style_loss.detach().item(),content_loss.detach().item()))

            loss = 1 * style_loss + 100 * content_loss

            loss.backward()
            opt.step()
            scheduler.step()

        if epoch % 200 == 0:
            test_true_y_list = []
            test_pred_y_list = []
            with torch.no_grad():
                generator.eval()
                discriminator.eval()
                for batch_idx, ((fake_data, fake_target),(real_data,real_target)) in enumerate(zip(fake_test_loader,real_test_loader)):

                    batch_len = np.min([fake_data.shape[0],real_data.shape[0]])
            
                    fake_data = fake_data[:batch_len].float().to(device)
                    fake_target = fake_target[:batch_len].long().to(device)
                    real_data = real_data[:batch_len].float().to(device)
                    real_target = real_target[:batch_len].long().to(device)

                    embedding = feature_extractor(fake_data)
                    fake_generated_data = generator(embedding)

                    fake_discr_val = discriminator(fake_generated_data)
                    fake_discr_loss = -1 * torch.sum(torch.log(1. - fake_discr_val), axis=0)
                    real_discr_val = discriminator(real_data)
                    real_discr_loss = -1 * torch.sum(torch.log(real_discr_val), axis=0)
                    style_loss = fake_discr_loss + real_discr_loss
                    
                    pred_class = class_classifier(feature_extractor(fake_generated_data))
                    # content_loss = ce_loss(pred_class, fake_target[:,0])
                    content_loss = torch.dist(feature_extractor(fake_generated_data),embedding)

                    pred_y = torch.argmax(pred_class.detach(), axis=1)

                    test_pred_y_list.extend(pred_y.tolist())
                    test_true_y_list.extend(fake_target[:,0].tolist())
                
                    test_acc = accuracy_score(test_true_y_list, test_pred_y_list)
                    test_rec = recall_score(test_true_y_list, test_pred_y_list,average='macro')
                    test_prc = precision_score(test_true_y_list, test_pred_y_list,average='macro')

                    print("[TEST EPOCH {%d}] style loss : {%.3f} content loss : {%.3f} acc : {%.3f} rec : {%.3f} prc : {%.3f}"%(epoch,style_loss.item(),content_loss.item(),test_acc,test_rec,test_prc))
