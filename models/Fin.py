from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn
import torch,math,logging,os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from .document_bert_architectures import DocumentBertLSTM, DocumentBertLinear, DocumentBertTransformer, DocumentBertMaxPool
from .document_bert_architectures_news import DocumentBertLSTMNews

from .functions import  DiffLoss, MSE, SIMSE, CMD
from sklearn.metrics import accuracy_score, matthews_corrcoef
from .SupCon import SupConLoss
# from .class_aware_sampler import ClassAwareSampler
# from .MyCustomDataset import MyCustomDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import pickle
import torch.nn.functional as F
import gzip



def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length=512):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.

    This is the input to any of the document bert architectures.

    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    # assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, 512), dtype=torch.long)
    document_seq_lengths = [] #number of sequence generated per document
    #Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == 512 and len(attention_masks) == 512 and len(input_type_ids) == 512

            #we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)








document_bert_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
    'DocumentBertTransformer': DocumentBertTransformer,
    'DocumentBertLinear': DocumentBertLinear,
    'DocumentBertMaxPool': DocumentBertMaxPool
}



# 定义神经网络模型
class Fin(nn.Module):
    def __init__(self, args=None, _END=True,input_dim=418):
        super(Fin, self).__init__()
        
        if args is not None:
            self.args = vars(args)
        print('labels',self.args['labels'])
        assert self.args['labels'] is not None, "Must specify all labels in prediction"
        
        self.finextract_fc1 = nn.Linear(input_dim, self.args['hid_fin'])  # 隐藏层
        self.finextract_fc2 = nn.Linear(self.args['hid_fin'], 1)  # 输出层

        self._END = _END
        self.log = logging.getLogger()

        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )
        for name, param in self.named_parameters():
            # if 'lstm' in name:
            #     param.requires_grad = False
            if (param.requires_grad):
                print(f"Parameter name: {name}")
        self.to("cuda:0")


    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = torch.sigmoid(self.fc2(x))
    #     return x


    def forward(self,  document_batch: torch.Tensor, 
                document_sequence_lengths: list, 
                device='cuda', fin=None, 
                news_batch=None, 
                news_sequence_lengths=None):
        
        # bert_output = torch.zeros(size=(fin.shape[0],
        #                                       min(fin.shape[1],self.bert_doc_classification.bert_batch_size),
        #                                       self.bert_doc_classification.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        
        # self.last_fin = last_fin = self.fc1(fin)
        # print('last_layer', last_layer) # 3* 768
        x = F.normalize(fin, dim=1)

        x = torch.relu(self.finextract_fc1(x))
        # print('x in FIN', x)
        if self._END:
            # print('x', x.shape)
            x = self.finextract_fc2(x)
            x = torch.sigmoid(x)
        
        return x
    


    def fit(self, train, dev, fin_train, fin_dev, news_train, news_dev,
            test, fin_test, news_test):
        print('START FIT ... ')
        """
        A list of
        :param documents: a list of documents
        :param labels: a list of label vectors
        :return:
        """

        train_documents, train_labels = train
        dev_documents, dev_labels = dev
        test_documents, test_labels = test
        del train
        del dev
        del test

        self.train()
  
        print(" encode start", )

  
        correct_output = torch.FloatTensor(train_labels)
        # print(correct_output)
        loss_weight = (1.0 / ( correct_output[:,1].sum() / correct_output.shape[0]) ).to(device=self.args['device'])
        print('$$$$$$$$$$$$loss_weight', loss_weight)
        # loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1).to(device=self.args['device'])

        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        # self.loss_function = CustomBCEWithPosWeightLoss(pos_weight=loss_weight)

  
        print("START EPOCH")
        correct_output = torch.FloatTensor(train_labels)
        
        ## collect loss for plot
        _all_loss = []
        _all_cls_loss = []
        _all_sim_loss = []
        _all_diff_loss = []
        _all_recon_loss = []
        _all_shared_loss = []
        _all_private_loss = []

        _all_loss_val = []
        _all_cls_loss_val = []
        _all_sim_loss_val = []
        _all_diff_loss_val = []
        _all_recon_loss_val = []
        _all_metric_microf1_val = []
        _all_metric_posf1_val = []
        _all_metric_mcc_val = []


        _all_loss_test = []
        _all_cls_loss_test = []
        _all_sim_loss_test = []
        _all_diff_loss_test = []
        _all_recon_loss_test = []
        _all_metric_microf1_test = []
        _all_metric_posf1_test = []
        _all_metric_mcc_test = []

        _all_epoch = 0

        for epoch in range(1,self.args['epochs']+1):
            _all_epoch += 1
            # print('EPOCH', epoch)
            # 
    
            # permutation = torch.randperm(document_representations.shape[0])
            # document_representations = document_representations[permutation]
            # document_sequence_lengths = document_sequence_lengths[permutation]
            # correct_output = correct_output[permutation]

            self.epoch = epoch
            epoch_loss = 0.0
            cmd_ = 0
            diff_ = 0
            recon_ = 0  
            n_ = 0
            cls_ = 0

            all_s = 0
            all_p = 0
           

            for i in range(0, fin_train.shape[0], self.args['batch_size']):
                self.optimizer.zero_grad()
                
                fin_tensors = torch.tensor(fin_train[i:i + self.args['batch_size']], dtype=torch.float32).to(device=self.args['device'])
        
        
                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                
                # print('fin_tensors', fin_tensors)
                # exit(0)
                batch_predictions = self.forward(None,
                                                None, self.args['device'],
                                                fin_tensors, 
                                                None, 
                                                None)

                # batch_predictions = batch_predictions.squeeze(dim=1)
                # batch_correct_output = batch_correct_output[:,1]

                loss = self.loss_function(batch_predictions.view(-1),
                                            batch_correct_output[:,1].view(-1))
                # print('pred', batch_predictions.view(-1))
                # print('corr', batch_correct_output[:,1].view(-1))
                # print('loss', (loss))
                cls_ += loss
                n_+=1
                _label = batch_correct_output[:,1].view(-1)

                epoch_loss += float(loss.item())
                
                #self.log.info(batch_predictions)
                loss.backward()
                torch.nn.utils.clip_grad_value_([param for param in self.parameters() if param.requires_grad], 1)
                # torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], max_norm=0.5)

                self.optimizer.step()
                self.optimizer.zero_grad()
         
    
            epoch_loss /= fin_train.shape[0] / self.args['batch_size']  # divide by number of batches per epoch
            print("=============== TRAIN ================== ")
            self.log.info('cls: %f', cls_ / n_)

            _all_cls_loss.append(cls_ / n_)

            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                self.tensorboard_writer.add_scalar('Loss/Train', epoch_loss, self.epoch)

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))
            _all_loss.append(epoch_loss)
            if epoch % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(os.path.join(self.args['model_directory'], "checkpoint_%s" % epoch),
                                     os.path.join(self.args['model_directory'], "newscheckpoint_%s" % epoch)
                                     )

            # evaluate on development data
            if epoch % self.args['evaluation_interval'] == 0:
                logging.info("======= EVAL  ======")
                _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc = self.predict((dev_documents, dev_labels), fin_dev, news_dev)
                _all_cls_loss_val.append(cls)
                _all_sim_loss_val.append(sim)
                _all_diff_loss_val.append(diff)
                _all_recon_loss_val.append(recon)
                _all_loss_val.append(_val_loss)
                _all_metric_microf1_val.append(mircof1)
                _all_metric_posf1_val.append(posf1)
                _all_metric_mcc_val.append(mcc)
                logging.info("======= TEST  ======")
                _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc = self.predict((test_documents, test_labels), fin_test, news_test)
                _all_cls_loss_test.append(cls)
                _all_sim_loss_test.append(sim)
                _all_diff_loss_test.append(diff)
                _all_recon_loss_test.append(recon)
                _all_loss_test.append(_val_loss)
                _all_metric_microf1_test.append(mircof1)
                _all_metric_posf1_test.append(posf1)
                _all_metric_mcc_test.append(mcc)
                ## vis shared subspace
                _path= os.path.join(self.args['model_directory'], "newscheckpoint_%s" % epoch)
                _labels = [i[1] for i in dev_labels]
                # self.model.tse_vis(_path,)# _labels)
            ## each epoch

            ## plot
                _path = os.path.join(self.args['model_directory'], "plot%s" % epoch)
                _PLOT = 1# Plot individual loss graphs
                
                # print('_all_diff_loss', _all_diff_loss)
                if _PLOT:
                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_loss).detach().cpu().numpy(), label='Train')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_loss_val).detach().cpu().numpy(), label='Val')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_loss_test).detach().cpu().numpy(), label='Test')

                    plt.xlabel('Epoch')
                    plt.ylabel('LOSS')
                    plt.legend()
                    plt.savefig(_path + 'ALL_LOSS.png')

                    plt.figure(figsize=(12, 6))

                    plt.subplot(2, 2, 1)
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_cls_loss).detach().cpu().numpy(), label='CLS Train Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_cls_loss_val).detach().cpu().numpy(), label='CLS Val Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_cls_loss_test).detach().cpu().numpy() , label='CLS Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('CLS Loss')
                    plt.legend()

                    plt.subplot(2, 2, 2)
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss).detach().cpu().numpy(), label='SIM Train Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss_val).detach().cpu().numpy(), label='SIM Val Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss_test).detach().cpu().numpy(), label='SIM Test Loss')

                    plt.xlabel('Epoch')
                    plt.ylabel('SIM Loss')
                    plt.legend()

                    plt.subplot(2, 2, 3)
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss).detach().cpu().numpy(), label='DIFF Train Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss_val).detach().cpu().numpy(), label='DIFF Val Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss_test).detach().cpu().numpy(), label='DIFF Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('DIFF Loss')
                    plt.legend()

                    plt.subplot(2, 2, 4)
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss).detach().cpu().numpy(), label='RECON Train Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss_val).detach().cpu().numpy(), label='RECON Val Loss')
                    # plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss_test).detach().cpu().numpy(), label='RECON Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('RECON Loss')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(_path + 'loss_plots.png')
                    # plt.show()

                    # Plot metrics in a single graph
                    plt.figure(figsize=(12, 6))

                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_microf1_val).detach().cpu().numpy(), label='Val Micro F1')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_microf1_test).detach().cpu().numpy(), label='Test Micro F1')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metrics')
                    plt.legend()
                    plt.savefig(_path + 'metric_microf1_plots.png')
                    # plt.show()

                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_posf1_val).detach().cpu().numpy(), label='Val Pos F1')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_posf1_test).detach().cpu().numpy(), label='Test Pos F1')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metrics')
                    plt.legend()
                    plt.savefig(_path + 'metric_posf1_plots.png')
                    # plt.show()

                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_mcc_val).detach().cpu().numpy(), label='Val Mcc')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_mcc_test).detach().cpu().numpy(), label='Test Mcc')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metrics')
                    plt.legend()
                    plt.savefig(_path + 'metric_mcc_plots.png')
                # plt.show()
                logging.info("==================================\n")


        ## for     
        ## test
        #        
        # self.predict((test_documents, test_labels), fin_test, news_test) 
    def predict(self, data, fin_dev, news_dev, threshold=0):
        """
        A tuple containing
        :param data:
        :return:
        """
        document_representations = None
        document_sequence_lengths = None
        news_representations = None
        news_sequence_lengths = None
        correct_output = None

        cls = 0
        sim =0
        diff =0 
        recon = 0
        mircof1 = 0
        posf1 = 0
        mcc = 0

        self.log.info('Evaluating on Epoch %i' % (self.epoch))

        # if isinstance(data, list):
        #     document_representations, document_sequence_lengths = encode_documents(data, self.bert_tokenizer)
        #     news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)

        # if isinstance(data, tuple) and len(data) == 2:
        #     self.log.info('Evaluating on Epoch %i' % (self.epoch))
        #     document_representations, document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer)
        #     news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)

        correct_output = torch.FloatTensor(data[1]).transpose(0,1)
        assert self.args['labels'] is not None
        self.eval()
    
        # print('correct_output', correct_output)
        with torch.no_grad():
            if True:
                predictions = torch.empty((fin_dev.shape[0], 1))
            else:
                predictions = torch.empty((fin_dev.shape[0], len(self.args['labels'])))

            epoch_loss = 0.0
            cmd_ = 0
            diff_ = 0
            recon_ = 0  
            n_ = 0
            cls_ = 0
            ## 
            for i in range(0, fin_dev.shape[0], self.args['batch_size']):
             
                fin_tensors = torch.tensor(fin_dev[i:i + self.args['batch_size']], dtype=torch.float32).to(device=self.args['device'])
              
                prediction = self.forward(None,
                                None,
                                    self.args['device'], 
                                    fin_tensors,
                                    None,
                                    None)
                # print('prediction', prediction)
                predictions[i:i + self.args['batch_size']] = prediction
                
                batch_correct_output = correct_output.transpose(0,1)[i:i + self.args['batch_size']].to(device=self.args['device'])
            
                # print('batch_correct_output', batch_correct_output.shape, correct_output.shape)
                # if self.args['use_sigmoid']:

                loss = self.loss_function(prediction.view(-1),
                                        batch_correct_output[:,1].view(-1))
            # else:
                #     loss = self.loss_function(prediction,
                #                            batch_correct_output)
                cls_ += loss
                n_ += 1
                # print('batch_correct_output', batch_correct_output.shape)
                _label = batch_correct_output[:,1].view(-1)
                # if not self.args['only_fusion']:
                #     cmd_loss = self.get_cmd_loss(_label)
                #     diff_loss = self.get_diff_loss(_label)
                #     recon_loss = self.get_recon_loss(_label)
                #     # cmd_loss = self.get_cmd_loss()
                #     # diff_loss = self.get_diff_loss()
                #     # recon_loss = self.get_recon_loss()

                #     cmd_ += cmd_loss
                #     diff_+= diff_loss
                #     recon_ += recon_loss
                #     # collect v6s
                #     _labels = [i for i in batch_correct_output[:,1]]
                #     self.model._coll(_labels)


            #############
        ## eval loss 
        logging.info('eval cls loss: %f', cls_ / n_)
        # logging.info('eval sim loss: %f', cmd_ / n_)
        # logging.info('eval diff loss: %f', diff_ / n_)
        # logging.info('eval recon loss: %f', recon_ / n_)
        cls = cls_ / n_
        # sim = cmd_ / n_
        # diff = diff_ / n_
        # recon = recon_ / n_
        # print('predictions',predictions )
        if True:
            scores = predictions
            # logging.info('score', scores)
            # 20 * 1
            predictions = torch.empty(fin_dev.shape[0], 2)

            for r in range(0, predictions.shape[0]):
                if scores[r] > 0.5:
                    predictions[r][1] = 1
                    predictions[r][0] = 0
                else:
                    predictions[r][1] = 0
                    predictions[r][0] = 1
            predictions = predictions.transpose(0, 1)

            # roc_auc = roc_auc_score(correct_output[1,:].view(-1), _sig)
            
            # 输出 ROC AUC
            roc_auc = 1.2
            # print("ROC AUC:", roc_auc)
            assert correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []

            for label_idx in range(predictions.shape[0]):
                correct = correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                print('correct', correct)
                print('predicted', predicted)

                if (label_idx==1):
                    _path = os.path.join(self.args['model_directory'], "data%s" % self.epoch)
                    with open(_path + 'correct.pkl', 'wb') as f:
                        pickle.dump(correct, f)
                    with open(_path + 'predicted.pkl', 'wb') as f:
                        pickle.dump(predicted, f)

                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)

                logging.info('F1\t%s\t%f' % (self.args['labels'][label_idx], present_f1_score))

                mcc_ = matthews_corrcoef(correct, predicted)  # 计算第一个标签的MCC
                micro_f1 = f1_score(correct,predicted, average='micro')
                macro_f1 = f1_score(correct,predicted,average='macro')

                accuracy = accuracy_score(correct,predicted)
                # 打印准确度和 Matthews相关系数
                logging.info('Accuracy: %f', accuracy)
                logging.info('Matthews Correlation Coefficient: %f', mcc_)
                logging.info('micro_f1: %f', micro_f1)
                logging.info('macro_f1: %f', macro_f1)
                logging.info('precision: %f', present_precision_score)
                logging.info('recall: %f', present_recall_score)
                if label_idx == 1:
                    mircof1 = micro_f1
                    posf1 = present_f1_score 
                    mcc = mcc_


            # print('correct_output', correct_output.shape, correct_output)
            # print('predictions', predictions.shape, predictions)


 
            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                for label_idx in range(predictions.shape[0]):
                    self.tensorboard_writer.add_scalar('Precision/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), precisions[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('Recall/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), recalls[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('F1/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), fmeasures[label_idx], self.epoch)
                self.tensorboard_writer.add_scalar('Micro-F1/Test', micro_f1, self.epoch)
                self.tensorboard_writer.add_scalar('Macro-F1/Test', macro_f1, self.epoch)
                self.tensorboard_writer.add_scalar('roc_auc/Test', roc_auc, self.epoch)

            with open(os.path.join(self.args['model_directory'], "eval_%s.csv" % self.epoch), 'w') as eval_results:
                eval_results.write('Metric\t' + '\t'.join([self.args['labels'][label_idx] for label_idx in range(predictions.shape[0])]) +'\n' )
                eval_results.write('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('F1\t' + '\t'.join([ str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Micro-F1\t' + str(micro_f1) + '\n' )
                eval_results.write('Macro-F1\t' + str(macro_f1) + '\n' )
                eval_results.write('Accuracy\t' + str(accuracy) + '\n' )
                eval_results.write('MCC\t' + str(mcc) + '\n' )

            ## eval clf loss, sim loss, diff loss, recons loss

        ## eval clf loss, sim loss, diff loss, recons loss

        self.train()
       
        
        _val_loss = cls + sim + diff + recon
        ## plot

        return _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc 
    

    def save_checkpoint(self, checkpoint_path: str, checkpoint_pathnews: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        else:
            raise ValueError("Attempting to save checkpoint to an existing directory")
        self.log.info("Saving checkpoint: %s" % checkpoint_path )

      
        torch.save(self.state_dict(), os.path.join(checkpoint_path, 'net1.pkl')  )


