from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn
import torch,math,logging,os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from .document_bert_architectures import DocumentBertLSTM, DocumentBertLinear, DocumentBertTransformer, DocumentBertMaxPool
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, confusion_matrix


from .functions import  *
from sklearn.metrics import accuracy_score, matthews_corrcoef
from .SupCon import SupConLoss
# from .class_aware_sampler import ClassAwareSampler
# from .MyCustomDataset import MyCustomDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import random

from torch.utils.data import DataLoader
from .OneText import DocumentBertLSTM, DocumentBertLinear, DocumentBertTransformer, DocumentBertMaxPool

import pandas as pd
import numpy as np
import pickle
import torch.nn.functional as F
import gzip

from torch.utils.data import WeightedRandomSampler

import torch
import torchkeras
import torch.nn.functional as F

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import time
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import numpy as np
# sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')

new_path = r'C:\Users\FxxkDatabase\Desktop\haochen\document-level-classification-main\document-level-classification-main'
sys.path.append(new_path)

from bert_document_classification.config import *
from bert_document_classification.prepare_data import load_data

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)


ngpu = 1

use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
print('*'*8, 'device:', device)

# 设置损失函数和评价指标
loss_func = torch.nn.CrossEntropyLoss()
metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
metric_name = 'acc'

df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])



#metric_func = lambda y_pred, y_true: recall_score(y_true, y_pred)
# metric_func = calculate_auc_or_return_zero

# 更改评估名称为 'auc'
#metric_name = 'auc'


def train_step(model, inps, labs, fin, optimizer):
    inps = inps.to(device)
    labs = labs.to(device)
    fin = fin.to(device)

    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零

    # forward
    logits = model(inps, fin)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)

    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy()) # 返回的是tensor还是标量？
    # print('*'*8, metric)
    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    return loss.item(), metric.item()



@torch.no_grad()
def validate_step(model, inps, labs, fin):
    inps = inps.to(device)
    labs = labs.to(device)

    fin = fin.to(device)

    model.eval()  # 设置eval mode

    # forward
    logits = model(inps, fin)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)

    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())  # 返回的是tensor还是标量？

# 将预测结果和真实标签转换为 numpy 数组
    pred_np = pred.cpu().numpy()
    labs_np = labs.cpu().numpy()

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(labs_np, pred_np)

    # 计算准确率、召回率和 F1 分数
    accuracy = accuracy_score(labs_np, pred_np)
    precision = precision_score(labs_np, pred_np)
    recall = recall_score(labs_np, pred_np)
    f1_macro = f1_score(labs_np, pred_np, average='macro')
    f1_micro = f1_score(labs_np, pred_np, average='micro')
    f1_label_1 = f1_score(labs_np, pred_np, average=None)[1]
    mcc = matthews_corrcoef(labs_np, pred_np)

    # 打印结果
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")
    print(f"F1 for Label 1: {f1_label_1}")
    print(f"MCC: {mcc}")
    print(f"Confusion Matrix:\n{conf_matrix}")


    return loss.item(), metric.item()

@torch.no_grad()
def validate_final(model, inps, labs, fin):
    inps = inps.to(device)
    labs = labs.to(device)
    fin = fin.to(device)

    model.eval()  # 设置eval mode

    # forward
    logits = model(inps, fin)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    pred = pred.cpu().numpy()
    labs = labs.cpu().numpy()
    logits= logits.cpu().numpy()

    ################
    # 将预测结果和真实标签转换为 numpy 数组
    pred_np = pred
    labs_np = labs

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(labs_np, pred_np)

    # 计算准确率、召回率和 F1 分数
    accuracy = accuracy_score(labs_np, pred_np)
    precision = precision_score(labs_np, pred_np)
    recall = recall_score(labs_np, pred_np)
    f1_macro = f1_score(labs_np, pred_np, average='macro')
    f1_micro = f1_score(labs_np, pred_np, average='micro')
    f1_label_1 = f1_score(labs_np, pred_np, average=None)[1]
    mcc = matthews_corrcoef(labs_np, pred_np)

    # 打印结果
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Micro: {f1_micro}")
    print(f"F1 for Label 1: {f1_label_1}")
    print(f"MCC: {mcc}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    try:
        auc = roc_auc_score(labs, logits[:, 1])
    except ValueError:
        auc = 0.5
    f1 = f1_score(labs, pred)
    recall = recall_score(labs, pred)
    precision = precision_score(labs, pred)
    accuracy = accuracy_score(labs, pred)

    return loss.item(), auc, f1, recall, precision, accuracy



def train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r=None, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(1, num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inps, labs, fin) in enumerate(train_dloader, start=1):
            loss, metric = train_step(model, inps, labs, fin, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inps, labs, fin) in enumerate(val_dloader, start=1):
            val_loss, val_metric = validate_step(model, inps, labs, fin)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        if scheduler_1r:
            scheduler_1r.step()

        # 记录和收集 1个epoch的训练和验证信息
        # columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name]
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
            record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # 保存最佳模型参数

        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            save_dir = r'C:\Users\FxxkDatabase\Desktop\haochen\goodnews1021\bert_document_classification-master\bert_document_classification-master\examples\save'
            # checkpoint = save_dir + '{:03d}_{:.3f}_ckpt.tar'.format(epoch, current_metric_avg) ############################################################
            checkpoint = save_dir + f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            # 保存
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)
    # 初始化存储验证结果的列表
    losses = []
    aucs = []
    f1_scores = []
    recalls = []
    precisions = []
    accuracies = []

    for val_step, (inps, labs, fin) in enumerate(val_dloader, start=1):
        loss, auc, f1, recall, precision, accuracy = validate_final(model, inps, labs, fin)
        
        # 将结果添加到列表中
        losses.append(loss)
        aucs.append(auc)
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)

    # 计算平均值
    average_loss = sum(losses) / len(losses)
    average_auc = sum(aucs) / len(aucs)
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_recall = sum(recalls) / len(recalls)
    average_precision = sum(precisions) / len(precisions)
    average_accuracy = sum(accuracies) / len(accuracies)

    # 打印平均结果
    print(f'Average Loss: {average_loss:.4f}')
    print(f'Average AUC: {average_auc:.4f}')
    print(f'Average F1 Score: {average_f1:.4f}')
    print(f'Average Recall: {average_recall:.4f}')
    print(f'Average Precision: {average_precision:.4f}')
    print(f'Average Accuracy: {average_accuracy:.4f}')

    endtime = time.time()
    time_elapsed = endtime - starttime
    print('*' * 27, 'training finished...')
    print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                            int((time_elapsed % 3600) // 60),
                                                            (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history


# 绘制训练曲线
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    imgs_dir = r'C:\Users\FxxkDatabase\Desktop\haochen\goodnews1021\bert_document_classification-master\bert_document_classification-master\examples\img'

    plt.savefig(imgs_dir + 'han_'+ metric + '.png')  # 保存图片
    plt.show()






class MyHAN(torch.nn.Module):
    def __init__(self, args, max_word_num, max_sents_num, 
                 vocab_size, hidden_size, num_classes, embedding_dim, embedding_matrix=None, dropout_p=0.5):
        super(MyHAN, self).__init__()

        if args is not None:
            self.args = vars(args)
        

        assert self.args['labels'] is not None, "Must specify all labels in prediction"
        self.log = logging.getLogger()
        HIDDEN_SIZE = self.args['hidden_dim']
        print('&&&&&&&&&&&&&&&&&&& HIDDEN_SIZE', HIDDEN_SIZE)
        self.dropout_rate = self.args['dr']
        self.use_sigmoid = self.args['use_sigmoid']



        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))


        ########################### imple
        self.max_word_num = max_word_num  # 15 句子所含最大词数
        self.max_sents_num = max_sents_num  # 60 文档所含最大句子数

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_id)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.dropout0 = torch.nn.Dropout(dropout_p)

        # self.layernorm1 = torch.nn.LayerNorm(normalized_shape=(sent_maxlen, embedding_dim), eps=1e-6)
        # self.layernorm2 = torch.nn.LayerNorm(normalized_shape=2*hidden_size, eps=1e-6)

        self.bi_rnn1 = torch.nn.GRU(self.embedding_dim, self.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.word_attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.word_ctx = torch.nn.Linear(self.hidden_size, 1, bias=False)

        self.bi_rnn2 = torch.nn.GRU(2 * self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.sent_attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.sent_ctx = torch.nn.Linear(self.hidden_size, 1, bias=False)

        self.dropout = torch.nn.Dropout(dropout_p)
        self.out = torch.nn.Linear(self.hidden_size * 2 + 406, self.num_classes)

        self.LR = 1e-2
        self.EPOCHS = 15
        
        
        params_to_update = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        self.optimizer = torch.optim.AdamW(params_to_update, lr=self.LR, weight_decay=1e-4)
        self.scheduler_1r = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch>EPOCHS*0.6 else 0.5 if epoch>EPOCHS*0.3 else 1)
    

        for name, param in self.named_parameters():
            if 'lstm' in name :#or 'trans' in name:
                param.requires_grad = False
            if 'finex' in name :
                param.requires_grad = False
            if (param.requires_grad):
                print(f"Parameter name: {name}")

        self.to("cuda:0")
    

    def forward(self,inputs = None,  finargs = None, document_batch=None, 
                document_sequence_lengths=None, 
                device='cuda:0', fin=None, 
                news_batch=None, news_sequence_lengths=None, 
                hidden1=None, hidden2=None):  # [b, 60, 15]
        # print('&&&&&&&&&&&&&&&&***&&&&&&(((&)))', inputs.shape)
        # print(finargs.shape) # 100*406
        embedded = self.dropout0(self.embedding(inputs))  # =>[b, 60, 15, 100]

        word_inputs = embedded.view(-1, embedded.size()[-2], embedded.size()[-1])  # =>[b*60, 15, embedding_dim]
        # word_inputs = self.layernorm1(word_inputs)
        self.bi_rnn1.flatten_parameters()
        """
        为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。
        类似我们调用tensor.contiguous
        """
        word_encoder_output, hidden1 = self.bi_rnn1(word_inputs,
                                                    hidden1)  # =>[b*60,15,2*hidden_size], [b*60,2,hidden_size]
        word_attn = self.word_attn(word_encoder_output).tanh()  # =>[b*60,15,hidden_size]
        word_attn_energy = self.word_ctx(word_attn)  # =>[b*60,15,1]
        word_attn_weights = F.softmax(word_attn_energy, dim=1).transpose(1, 2)  # =>[b*60,15,1]=>[b*60,1,15]
        word_att_level_output = torch.bmm(word_attn_weights, word_encoder_output)  # =>[b*60,1,2*hidden_size]

        sent_inputs = word_att_level_output.squeeze(1).view(-1, self.max_sents_num,
                                                            2 * self.hidden_size)  # =>[b*60,2*hidden_size]=>[b,60,2*hidden_size]
        self.bi_rnn2.flatten_parameters()
        sent_encoder_output, hidden2 = self.bi_rnn2(sent_inputs, hidden2)  # =>[b,60,2*hidden_size], [b,2,hidden_size]
        sent_attn = self.sent_attn(sent_encoder_output).tanh()  # =>[b,60,hidden_size]
        sent_attn_energy = self.sent_ctx(sent_attn)  # =>[b,60,1]
        sent_attn_weights = F.softmax(sent_attn_energy, dim=1).transpose(1, 2)  # =>[b,60,1]=>[b,1,60]
        sent_att_level_output = torch.bmm(sent_attn_weights, sent_encoder_output)  # =>[b,1,2*hidden_size]

        # logits = self.out(self.dropout(self.layernorm2(sent_att_level_output.squeeze(1))))  # =>[b,2*hidden_size]=>[b,num_classes]
        a = self.dropout(sent_att_level_output.squeeze(1))
        concatenated_tensor = torch.cat((a, finargs), dim=1)
        logits = self.out(
            concatenated_tensor
            )  # =>[b,2*hidden_size]=>[b,num_classes]
        return logits  # [b,num_classes]


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
    

        self.train()

      

        correct_output = torch.FloatTensor(train_labels)

        if self.args['use_sigmoid']:
            loss_weight = (1.0 / ( correct_output[:,1].sum() / correct_output.shape[0]) ).to(device=self.args['device'])

        else:
            loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1).to(device=self.args['device'])
     
        if self.args['sampler'] == 0:
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        else:
            # print("&&&&&&&&&&&&&&&&&&&   no weight")
            self.loss_function = torch.nn.BCEWithLogitsLoss()#pos_weight=loss_weight)
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
        _all_metric_macro_val = []

        _all_loss_test = []
        _all_cls_loss_test = []
        _all_sim_loss_test = []
        _all_diff_loss_test = []
        _all_recon_loss_test = []
        _all_metric_microf1_test = []
        _all_metric_posf1_test = []
        _all_metric_mcc_test = []
        _all_metric_macro_test = []

        _all_epoch = 0

        # indices_label_1 = torch.nonzero(correct_output[:, 1].view(-1) == 1).squeeze()

        # # 使用索引来提取相应的数据
        # document_representations_label_1 = document_representations[indices_label_1]
        # document_sequence_lengths_label_1 = document_sequence_lengths[indices_label_1]
        # news_representations_label_1 = news_representations[indices_label_1]
        # news_sequence_lengths_label_1 = news_sequence_lengths[indices_label_1]
        # fin_train_label_1 = fin_train[indices_label_1]
        # correct_output_label_1 = correct_output[indices_label_1]
        # print('correct_output_label_1', correct_output_label_1.shape)

        # // load data_base_dir
    
        for epoch in range(1,self.args['epochs']+1):
            _all_epoch += 1
            # print('EPOCH', epoch)
            # 
            # 获取数据集的大小
            # data_size = document_representations.shape[0]

            
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
            _bala = 0
           


            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                self.optimizer.zero_grad()
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_sequence_lengths= document_sequence_lengths[i:i+self.args['batch_size']]
                #self.log.info(batch_document_tensors.shape)
                
                batch_news_tensors = news_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_news_sequence_lengths= news_sequence_lengths[i:i+self.args['batch_size']]
              

                fin_tensors = torch.tensor(fin_train[i:i + self.args['batch_size']], dtype=torch.float32).to(device=self.args['device'])
        
                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                #batch_correct_output_con = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                
          

                # print('fin_tensors', fin_tensors)
                # exit(0)
                batch_predictions = self.forward(batch_document_tensors,
                                                batch_document_sequence_lengths, self.args['device'],
                                                fin_tensors, 
                                                batch_news_tensors, 
                                                batch_news_sequence_lengths)

                # batch_predictions = batch_predictions.squeeze(dim=1)
                # batch_correct_output = batch_correct_output[:,1]

                ### contrast first


                if self.args['use_sigmoid']:
                    # print('batch_correct_output', batch_predictions.view(-1).shape, batch_correct_output)
                    # print('batch_predictions', batch_correct_output[:,1].view(-1).shape,  batch_predictions)

                    loss = self.loss_function(batch_predictions.view(-1),
                                            batch_correct_output[:,1].view(-1))
                else:
                    loss = self.loss_function(batch_predictions,
                                            batch_correct_output)
                
                # print('loss', (loss))
                cls_ += loss

                _label = batch_correct_output[:,1].view(-1)

                
                n_ += 1

                

                epoch_loss += float(loss.item())
                
                #self.log.info(batch_predictions)
                loss.backward()
                torch.nn.utils.clip_grad_value_([param for param in self.parameters() if param.requires_grad], 1)
                # torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], max_norm=0.5)

                self.optimizer.step()
                self.optimizer.zero_grad()

                
        
            epoch_loss /= document_representations.shape[0] / self.args['batch_size']  # divide by number of batches per epoch
            print("=============== TRAIN ================== ")
           

            self.log.info('bala: %f', _bala / n_)

            self.log.info('cls: %f', cls_ / n_)
            self.log.info('diff: %f', diff_ / n_)
            self.log.info('sim: %f', cmd_ / n_)
            self.log.info('recon: %f', recon_ / n_)
            _all_cls_loss.append(cls_ / n_)
            
            self.log.info('shared: %f', all_s / n_)
            self.log.info('private: %f', all_p / n_)
  

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
                _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc, macro = \
                self.predict((dev_documents, dev_labels), fin_dev, news_dev, if_eval=1)
                _all_cls_loss_val.append(cls)
                _all_sim_loss_val.append(sim)
                _all_diff_loss_val.append(diff)
                _all_recon_loss_val.append(recon)
                _all_loss_val.append(_val_loss)
                _all_metric_microf1_val.append(mircof1)
                _all_metric_posf1_val.append(posf1)
                _all_metric_mcc_val.append(mcc)
                _all_metric_macro_val.append(macro)
                logging.info("======= TEST  ======")
                _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc, macro = \
                    self.predict((test_documents, test_labels), fin_test, news_test, if_eval=0)
                _all_cls_loss_test.append(cls)
                _all_sim_loss_test.append(sim)
                _all_diff_loss_test.append(diff)
                _all_recon_loss_test.append(recon)
                _all_loss_test.append(_val_loss)
                _all_metric_microf1_test.append(mircof1)
                _all_metric_posf1_test.append(posf1)
                _all_metric_mcc_test.append(mcc)
                _all_metric_macro_test.append(macro)
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
                    plt.close()

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
                    plt.close()

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
                    plt.close()

                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_mcc_val).detach().cpu().numpy(), label='Val Mcc')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_mcc_test).detach().cpu().numpy(), label='Test Mcc')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metrics')
                    plt.legend()
                    plt.savefig(_path + 'metric_mcc_plots.png')
                    plt.close()
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_macro_val).detach().cpu().numpy(), label='Val Mcc')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_metric_macro_test).detach().cpu().numpy(), label='Test Mcc')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metrics')
                    plt.legend()
                    plt.savefig(_path + 'metric_macro_plots.png')
                    plt.close()
                # plt.show()
                logging.info("==================================\n")


        ## for     
        ## test
        #        
        # self.predict((test_documents, test_labels), fin_test, news_test) 
    def predict(self, data, fin_dev, news_dev, threshold=0, if_eval = 1):
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


        if isinstance(data, list):
            document_representations, document_sequence_lengths = encode_documents(data, self.bert_tokenizer)
            news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)

        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            # document_representations, document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer)
            # news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)
            if if_eval:
                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_representations_eval.pkl', 'wb') as file:
                #     pickle.dump(document_representations, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_sequence_lengths_eval.pkl', 'wb') as file:
                #     pickle.dump(document_sequence_lengths, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_representations_eval.pkl', 'wb') as file:
                #     pickle.dump(news_representations, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_sequence_lengths_eval.pkl', 'wb') as file:
                #     pickle.dump(news_sequence_lengths, file)
################################
                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_representations_eval.pkl', 'rb') as file:
                    document_representations = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_sequence_lengths_eval.pkl', 'rb') as file:
                    document_sequence_lengths = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_representations_eval.pkl', 'rb') as file:
                    news_representations = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_sequence_lengths_eval.pkl', 'rb') as file:
                    news_sequence_lengths = pickle.load(file)
            else:############################
                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_representations_test.pkl', 'wb') as file:
                #     pickle.dump(document_representations, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_sequence_lengths_test.pkl', 'wb') as file:
                #     pickle.dump(document_sequence_lengths, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_representations_test.pkl', 'wb') as file:
                #     pickle.dump(news_representations, file)

                # with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_sequence_lengths_test.pkl', 'wb') as file:
                #     pickle.dump(news_sequence_lengths, file)
################################
                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_representations_test.pkl', 'rb') as file:
                    document_representations = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_sequence_lengths_test.pkl', 'rb') as file:
                    document_sequence_lengths = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_representations_test.pkl', 'rb') as file:
                    news_representations = pickle.load(file)

                with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_sequence_lengths_test.pkl', 'rb') as file:
                    news_sequence_lengths = pickle.load(file)

#########################33333333333333333333

            correct_output = torch.FloatTensor(data[1]).transpose(0,1)
            assert self.args['labels'] is not None

        self.eval()
        # print('correct_output', correct_output)
        with torch.no_grad():
            if self.args['use_sigmoid']:
                predictions = torch.empty((document_representations.shape[0], 1))
            else:
                predictions = torch.empty((document_representations.shape[0], len(self.args['labels'])))

            epoch_loss = 0.0
            cmd_ = 0
            diff_ = 0
            recon_ = 0  
            n_ = 0
            cls_ = 0

   
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_sequence_lengths= document_sequence_lengths[i:i+self.args['batch_size']]

                fin_tensors = torch.tensor(fin_dev[i:i + self.args['batch_size']], dtype=torch.float32).to(device=self.args['device'])
                batch_news_tensors = news_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_news_sequence_lengths= news_sequence_lengths[i:i+self.args['batch_size']]
             
                prediction = self.forward(batch_document_tensors,
                                                          batch_document_sequence_lengths,
                                                          device=self.args['device'], 
                                                          fin=fin_tensors,
                                                          news_batch=batch_news_tensors,
                                                          news_sequence_lengths = batch_news_sequence_lengths)
                # print('prediction', prediction)
                predictions[i:i + self.args['batch_size']] = prediction
                
                batch_correct_output = correct_output.transpose(0,1)[i:i + self.args['batch_size']].to(device=self.args['device'])
            
                # print('batch_correct_output', batch_correct_output.shape, correct_output.shape)
                if self.args['use_sigmoid']:

                    loss = self.loss_function(prediction.view(-1),
                                           batch_correct_output[:,1].view(-1))
                else:
                    loss = self.loss_function(prediction,
                                           batch_correct_output)
                cls_ += loss
                n_ += 1
                # print('batch_correct_output', batch_correct_output.shape)
                _label = batch_correct_output[:,1].view(-1)
                
          

            #############
        ## eval loss 
        logging.info('eval cls loss: %f', cls_ / n_)
        logging.info('eval sim loss: %f', cmd_ / n_)
        logging.info('eval diff loss: %f', diff_ / n_)
        logging.info('eval recon loss: %f', recon_ / n_)
        cls = cls_ / n_
        sim = cmd_ / n_
        diff = diff_ / n_
        recon = recon_ / n_
   
       

        if self.args['use_sigmoid']:
            scores = predictions
            logging.info('score', scores)
            # 20 * 1
            predictions = torch.empty(document_representations.shape[0], 2)

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
                weighted_f1 = f1_score(correct,predicted,average='weighted')

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
                    macrof1 = macro_f1


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


        else:
            # print('scores', predictions)  # 20* 1 , 
            scores = predictions
            # 20 * 2 
            predictions = torch.empty(document_representations.shape[0], 2)

            for r in range(0, predictions.shape[0]):
                for c in range(0, predictions.shape[1]):
                    if scores[r][c] > threshold:
                        predictions[r][c] = 1
                    else:
                        predictions[r][c] = 0
            predictions = predictions.transpose(0, 1)

            # print('predictions    ', predictions) 
            # print('correct_output', correct_output)

            if correct_output is None:
                return predictions.cpu()
            else:

                # 使用 roc_auc_score 函数计算 ROC AUC
                # print(correct_output[1,:].view(-1).shape)
                # print(scores[:,1].view(-1).shape)
                # roc_auc = roc_auc_score(correct_output[1,:].view(-1), scores.view(-1))
                _s = scores[:,1].view(-1)
                original_value = torch.atanh(_s)
                _sig = 1 / (1 + np.exp(-original_value))
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
                        # with open(_path + 'correct.pkl', 'wb') as f:
                        #     pickle.dump(correct, f)
                        # with open(_path + 'predicted.pkl', 'wb') as f:
                        #     pickle.dump(predicted, f)


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
                    weighted_f1 = f1_score(correct,predicted,average='weighted')

                    accuracy = accuracy_score(correct,predicted)
                    # 打印准确度和 Matthews相关系数
                    logging.info('Accuracy: %f', accuracy)
                    logging.info('Matthews Correlation Coefficient: %f', mcc_)
                    logging.info('micro_f1: %f', micro_f1)
                    logging.info('macro_f1: %f', macro_f1)

                    logging.info('weighted_f1: %f', weighted_f1)

                    if label_idx == 1:
                        mircof1 = micro_f1
                        posf1 = present_f1_score 
                        mcc = mcc_
                        macrof1 = macro_f1
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


        self.train()
        # self.bert_doc_classification.document_bertlstm_news.train()
        
        _val_loss = cls + sim + diff + recon
        ## plot

        return _val_loss, cls, sim, diff, recon, mircof1, posf1, mcc , macrof1
    

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

        # #save finetune parameters
        # net = self.bert_doc_classification
        # if isinstance(self.bert_doc_classification, nn.DataParallel):
        #     net = self.bert_doc_classification.module
        # torch.save(net.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        # #save configurations
        # net.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        # #save exact vocabulary utilized
        # self.bert_tokenizer.save_vocabulary(checkpoint_path)

        # if not os.path.exists(checkpoint_pathnews):
        #     os.mkdir(checkpoint_pathnews)
        # else:
        #     raise ValueError("Attempting to save checkpoint to an existing directory")
        # self.log.info("Saving checkpoint: %s" % checkpoint_pathnews )

        # net = self.model.document_bertlstm_news
        # if isinstance(self.model.document_bertlstm_news, nn.DataParallel):
        #     net = self.model.document_bertlstm_news.module
        # torch.save(net.state_dict(), os.path.join(checkpoint_pathnews, WEIGHTS_NAME))
        # #save configurations
        # net.config.to_json_file(os.path.join(checkpoint_pathnews, CONFIG_NAME))
        # #save exact vocabulary utilized
        # self.bert_tokenizer.save_vocabulary(checkpoint_pathnews)





















