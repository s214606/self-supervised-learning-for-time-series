import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from Loss_functions import *

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def Trainer(model,  temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, model_F=None, model_F_optimizer=None,
            classifier=None, classifier_optimizer=None):
    print("Training started!")
    criterion = nn.CrossEntropyLoss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    os.makedirs(os.path.join("Loss values"), exist_ok=True)

    """Pre-training for pre-training a model on labelled data, decided by the "training_mode" parameter."""
    if training_mode == "pre_train":
        for epoch in range(1, config.num_epoch + 1):
            print(f"Started pre-training Epoch {epoch}.")
            train_loss, train_acc, train_auc, total_loss_c_pre, total_loss_f_pre, total_loss_t_pre = model_pretrain(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                                              train_dl, config, device, training_mode, model_F = model_F,
                                                              model_F_optimizer = model_F_optimizer)
            
            print(f'\nPre-training Epoch : {epoch} of {config.num_epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\t | \tTrain AUC : {train_auc:2.4f}\n'
                         )
        # Save the model
        os.makedirs(os.path.join("Saved models"), exist_ok=True) # only save in self_supervised mode.
        chkpoint = {'model_state_dict': model.state_dict(),}
        torch.save(chkpoint, os.path.join("Saved models", f'ckp_last.pt'))
        print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))
    
        print("Training finished.")
        #Save loss values
        total_loss_c_pre = torch.tensor(total_loss_c_pre)
        total_loss_f_pre = torch.tensor(total_loss_f_pre)
        total_loss_t_pre = torch.tensor(total_loss_t_pre)
        torch.save(total_loss_c_pre, os.path.join("Loss values", "total_loss_c_pre.pt"))
        torch.save(total_loss_f_pre, os.path.join("Loss values", "total_loss_f_pre.pt"))
        torch.save(total_loss_t_pre, os.path.join("Loss values", "total_loss_t_pre.pt"))

    """Fine-tuning and test"""
    if training_mode != "pre_train":
        print("Fine-tuning started!")
        performance_list = []
        total_f1 = []
        for epoch in range(1, config.num_epoch + 1):
            valid_loss, valid_acc, valid_auc, valid_prc, emb_finetune, label_finetune, F1, total_loss_c_fine, total_loss_f_fine, total_loss_t_fine = model_finetune(model, temporal_contr_model,
                                                            valid_dl, config, device, training_mode, model_optimizer,
                                                            model_F = model_F, model_F_optimizer = model_F_optimizer,
                                                            classifier = classifier, classifier_optimizer = classifier_optimizer)
            print(f'\nEpoch : {epoch}\n'
                         f'Finetune Loss  : {valid_loss:.4f}\t | \tFinetune Accuracy : {valid_acc:2.4f}\t | '
                         f'\tFinetune AUC : {valid_auc:2.4f} \t |Finetune PRC: {valid_prc:0.4f} ')
        # Test current epoch
            test_loss, test_acc, test_auc, test_prc, emb_test, label_test, performance = model_test(model,
                                                    test_dl, config, device, training_mode, classifier = classifier)
            performance_list.append(performance)
    
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:,0], axis=0)]
        print('Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | PRC=%.4f'
            % (best_performance[0], best_performance[1], best_performance[2], best_performance[3], best_performance[4], best_performance[5]))
        # Save loss values
        total_loss_c_fine = torch.tensor(total_loss_c_fine)
        total_loss_f_fine = torch.tensor(total_loss_f_fine)
        total_loss_t_fine = torch.tensor(total_loss_t_fine)
        best_performance = torch.tensor(best_performance)
        torch.save(best_performance, os.path.join("Loss values", "best_performance.pt"))
        torch.save(total_loss_c_fine, os.path.join("Loss values", "total_loss_c_fine.pt"))
        torch.save(total_loss_f_fine, os.path.join("Loss values", "total_loss_f_fine.pt"))
        torch.save(total_loss_t_fine, os.path.join("Loss values", "total_loss_t_fine.pt"))
        """
        # train classifier: KNN
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(emb_finetune.detach().cpu().numpy(), label_finetune)
        knn_acc_train = neigh.score(emb_finetune.detach().cpu().numpy(), label_finetune)
        print('KNN finetune acc:', knn_acc_train)
        # test the downstream classifier
        representation_test = emb_test.detach().cpu().numpy()
        knn_result = neigh.predict(representation_test)
        knn_result_score = neigh.predict_proba(representation_test)
        one_hot_label_test = one_hot_encoding(label_test)
        print(classification_report(label_test, knn_result, digits=4))
        print(confusion_matrix(label_test, knn_result))
        knn_acc = accuracy_score(label_test, knn_result)
        precision = precision_score(label_test, knn_result, average='macro', )
        recall = recall_score(label_test, knn_result, average='macro', )
        F1 = f1_score(label_test, knn_result, average='macro')
        auc = roc_auc_score(knn_result_score, one_hot_label_test, average="macro", multi_class="ovr")
        prc = average_precision_score(knn_result_score, one_hot_label_test, average="macro")
        print("KNN Train Acc:{}. '\n' Test: acc {}, precision {}, Recall {}, F1 {}, AUROC {}, AUPRC {}"
            "".format(knn_acc_train, knn_acc, precision, recall, F1, auc, prc))
        """

def model_pretrain(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_loader, config,
                   device, training_mode, criterion = None, model_F = None, model_F_optimizer = None, delta = 1):
    """Pre-trains a model using the given data given in train_loader (generated by the data_generator), and trains
    model weights on this."""
    total_loss = []
    total_acc = []
    total_auc = []
    model.train()
    total_loss_t = []
    total_loss_f = []
    total_loss_c = []

    # Compute pre-train loss using using NTXentLoss from loss_functions.py
    nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                        config.Context_Cont.use_cosine_similarity)

    i = 0
    for batch_idx, (data, labels, data_aug, data_f, data_f_aug) in enumerate(train_loader):
        i += 1
        data, labels = data.float().to(device), labels.long().to(device)
        data_aug = data_aug.float().to(device)
        data_f, data_f_aug = data_f.float().to(device), data_f_aug.float().to(device)

        # Optimizer
        model_optimizer.zero_grad()

        # Produce embeddings of the data using the model's forward method (simply by calling the model)
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(data_aug, data_f_aug)
        
        # Compute the losses for time and frequency domain
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        total_loss_t.append(loss_t)
        total_loss_f.append(loss_f)
        

        # Compute the loss for the time-frequency space
        loss_TF = nt_xent_criterion(z_t, z_f)
        loss_TF1, loss_TF2, loss_TF3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)

        # Compute the consistency loss from the losses of the pairs of augmentations and originals
        # This is done using equation 3 from the paper, with delta as a parameter.
        loss_c = (loss_TF - loss_TF1 + delta) + (loss_TF - loss_TF2 + delta) + (loss_TF - loss_TF3 + delta)
        total_loss_c.append(loss_c)


        # Compute the total loss
        lam = 0.2
        loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        print(f"Finished optimizing batch {batch_idx}.")

        # Terminate early if a certain threshold is reached (for testing compilation)
        terminate_threshold = 999999
        if i > terminate_threshold:
            break

    # Print the results
    print('pretraining: overall loss: {}, l_t: {}, l_f:{}, l_c:{}'.format(loss,loss_t,loss_f, loss_c))

    total_loss = torch.tensor(total_loss).mean()
    if training_mode == "pre_train":
        total_acc = 0
        total_auc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
    return total_loss, total_acc, total_auc, total_loss_c,total_loss_f, total_loss_t

def model_finetune(model, temporal_contr_model, val_dl, config, device, training_mode, model_optimizer, 
                   model_F = None, model_F_optimizer = None, classifier = None, classifier_optimizer = None):
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_prc = []
    total_loss_c = []
    total_loss_f = []
    total_loss_t = []


    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    i = 0
    for data, labels, aug1, data_f, aug1_f in val_dl:
        i += 1
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)
    
        """Produce embeddings"""
        h_t, z_t, h_f, z_f=model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug=model(aug1, aug1_f)

        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        total_loss_t.append(loss_t)
        total_loss_f.append(loss_f)

        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                            z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3) #
        total_loss_c.append(loss_c)

        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels) # predictor loss, actually, here is training loss

        lam = 0.2
        loss =  loss_p + (1-lam)*loss_c + lam*(loss_t + loss_f )

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = predictions.detach().cpu().numpy()

        auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
        prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)

        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_prc.append(prc_bs)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        if training_mode != "pre_train":
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
        
        print(f"Finished optimizing batch {i}.")
        terminate_threshold = 999999
        if i > terminate_threshold:
            break


    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    precision = precision_score(labels_numpy, pred_numpy, average='macro', zero_division=False)  # labels=np.unique(ypred))
    recall = recall_score(labels_numpy, pred_numpy, average='macro', zero_division=False)  # labels=np.unique(ypred))
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', zero_division=False)  # labels=np.unique(ypred))
    print('Testing: Precision = %.4f | Recall = %.4f | F1 = %.4f' % (precision * 100, recall * 100, F1 * 100))

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()  # average acc
    total_prc = torch.tensor(total_prc).mean()

    return total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs, F1, total_loss_c, total_loss_f, total_loss_t

def model_test(model, test_data, config, device, training_mode,
               classifier = None):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _, data_f, _ in test_data:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            # Make predictions
            h_t, z_t, h_f, z_f = model(data, data_f)

            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat) 
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
                
            # Make auc score 0 if no values match
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                               average="macro", multi_class="ovr")
            except:
                auc_bs = np.float32(0)
            # Is reshaping a proper solution??
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy().reshape(-1), pred_numpy.reshape(-1), average="macro")

            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=False)
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=False)
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=False)
    acc = accuracy_score(labels_numpy_all, pred_numpy_all)

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    # precision_mean = torch.tensor(total_precision).mean()
    # recal_mean = torch.tensor(total_recall).mean()
    # f1_mean = torch.tensor(total_f1).mean()
    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    print('Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | PRC=%.4f'
          % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))

    emb_test_all = torch.concat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance