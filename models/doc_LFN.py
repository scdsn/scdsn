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

from bert_document_classification.Fin import Fin
from bert_document_classification.doc_onetext import BertForDocumentClassification_ONLY
document_bert_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
    'DocumentBertTransformer': DocumentBertTransformer,
    'DocumentBertLinear': DocumentBertLinear,
    'DocumentBertMaxPool': DocumentBertMaxPool
}
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







class LFN(nn.Module):
    def __init__(self,args=None):
        super(LFN, self).__init__()


        if args is not None:
            self.args = vars(args)
        

        assert self.args['labels'] is not None, "Must specify all labels in prediction"
        self.log = logging.getLogger()
        HIDDEN_SIZE = self.args['hidden_dim']
        print('&&&&&&&&&&&&&&&&&&& HIDDEN_SIZE', HIDDEN_SIZE)
        self.dropout_rate = self.args['dr']
        self.use_sigmoid = self.args['use_sigmoid']
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])
        #account for some random tensorflow naming scheme
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])
        config.__setattr__('num_labels',len(self.args['labels']))
        config.__setattr__('bert_batch_size',self.args['bert_batch_size'])


########################
        self.bertnews_tokenizer = BertTokenizer.from_pretrained(self.args['newsbert_model_path'])
        #account for some random tensorflow naming scheme
        if os.path.exists(self.args['newsbert_model_path']):
            if os.path.exists(os.path.join(self.args['newsbert_model_path'], CONFIG_NAME)):
                newsconfig = BertConfig.from_json_file(os.path.join(self.args['newsbert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['newsbert_model_path'], 'bert_config.json')):
                newsconfig = BertConfig.from_json_file(os.path.join(self.args['newsbert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            newsconfig = BertConfig.from_pretrained(self.args['newsbert_model_path'])
        newsconfig.__setattr__('num_labels',len(self.args['labels']))
        newsconfig.__setattr__('bert_batch_size',self.args['bert_batch_size'])

######################

        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))


        self.mda = DocumentBertLSTM.from_pretrained(self.args['bert_model_path'], config=config)
        self.mda.freeze_bert_encoder()
        # self.mda.unfreeze_bert_encoder_last_layers()
        self.mda._END = False

        self.news = document_bert_architectures[self.args['architecture']].from_pretrained(self.args['newsbert_model_path'], config=newsconfig)
        self.news.freeze_bert_encoder()
        # self.news.unfreeze_bert_encoder_last_layers()
        self.news._END = False

        self.fin = Fin(args, False)



        # encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # self.fusion = nn.Sequential() # only fusion
        # self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        # print('dropout_rate', self.dropout_rate)
        # if self.use_sigmoid:
        #     self.fusion.add_module('fusion_layer_2_', nn.Linear(self.args['hid_fin']#+self.args['hid_trans']
        #                                                         , 1))
        #     self.fusion.add_module('fusion_layer_3', nn.Sigmoid())  
        # else:
        # # temp = bert_model_config.hidden_size // 2
        #     self.fusion.add_module('fusion_layer_2_', nn.Linear(self.args['hid_fin']+self.args['hid_trans']
        #                                                         , 2))
        #     self.fusion.add_module('fusion_layer_3', nn.Tanh())

        ## transfomrer
        self.mdafc = nn.Sequential(
                    nn.Linear(768, HIDDEN_SIZE),
                    nn.ReLU()
                )
        self.newsfc = nn.Sequential(
                    nn.Linear(768, HIDDEN_SIZE),
                    nn.ReLU()
                )
        
       
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )
        
        for name, param in self.named_parameters():
            if 'lstm' in name :#or 'trans' in name:
                param.requires_grad = False
            if 'finex' in name :
                param.requires_grad = False
            if (param.requires_grad):
                print(f"Parameter name: {name}")

        self.to("cuda:0")

    def forward(self,document_batch: torch.Tensor, 
                document_sequence_lengths: list, 
                device='cuda:0', fin=None, 
                news_batch=None, news_sequence_lengths=None):
        last_layer = self.mda(document_batch, document_sequence_lengths, device)
        last_fin = self.fin(None, None, self.args['device'], fin, None, None)
        last_layer_news = self.news(news_batch, news_sequence_lengths, device)
        
        last_layer = self.mdafc(last_layer)
        last_layer_news = self.newsfc(last_layer_news)
        return self._fusion(last_layer, last_fin,last_layer_news )
    
    def _fusion(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
      
        output = self.sigmoid(output)
        return output

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
    

        self.fin.train()
        self.mda.train()
        self.news.train()

        print(" encode start", )
        if self.args['encoding']==1:
            document_representations, document_sequence_lengths  = encode_documents(train_documents, self.bert_tokenizer)
            news_representations, news_sequence_lengths  = encode_documents(news_train, self.bertnews_tokenizer)
  

        else:
            with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_representations.pkl', 'rb') as file:
                document_representations = pickle.load(file)

            with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\document_sequence_lengths.pkl', 'rb') as file:
                document_sequence_lengths = pickle.load(file)

            with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_representations.pkl', 'rb') as file:
                news_representations = pickle.load(file)

            with open(r'C:\Users\FxxkDatabase\Desktop\haochen\data\news_sequence_lengths.pkl', 'rb') as file:
                news_sequence_lengths = pickle.load(file)

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

        print('document_representations.shape[0]', document_representations.shape[0])
        assert document_representations.shape[0] == correct_output.shape[0]
        assert news_representations.shape[0] == correct_output.shape[0]

        # if torch.cuda.device_count() > 1:
        #     self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.mda.to(device=self.args['device'])
        self.news.to(device=self.args['device'])
        self.fin.to(device=self.args['device'])

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


        for epoch in range(1,self.args['epochs']+1):
            _all_epoch += 1
            # print('EPOCH', epoch)
            # 
            # 获取数据集的大小
            data_size = document_representations.shape[0]

            
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
            # 使用 Class-Aware Sampler
            
            # batch_indices = list(sampler)
            # for i in range(0, len(batch_indices), self.args['batch_size']):
            #     self.optimizer.zero_grad()
            #     batch_document_tensors = document_representations[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])
            #     batch_document_sequence_lengths = document_sequence_lengths[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])

            #     batch_news_tensors = news_representations[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])
            #     batch_news_sequence_lengths = news_sequence_lengths[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])

            #     fin_tensors = torch.tensor(fin_train[batch_indices[i:i + self.args['batch_size']]], dtype=torch.float32).to(device=self.args['device'])
            #     batch_correct_output = correct_output[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])
            #     batch_correct_output_con = correct_output[batch_indices[i:i + self.args['batch_size']]].to(device=self.args['device'])
                
                # 其他批次处理的代码

            all_hidden_out = torch.empty(0, 768).to(self.args['device'])  #
            all_labels = torch.empty(0).to(self.args['device'])


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
                
          
                if self.args['sampler']:
                    # print("&&&&&&&&&&&&&& sampler!")
                    batch_correct_output[:,1].view(-1) #中找出值为 0 的样本的索引
                    indices_to_replace = torch.where(batch_correct_output[:, 1].view(-1) == 0)[0].tolist()
                    flatten_second_column = batch_correct_output[:, 1].view(-1)

                    # 计算值为 0 和 1 的数量
                    num_zeros = torch.sum(flatten_second_column == 0).item()
                    num_ones = torch.sum(flatten_second_column == 1).item()
                    # 找到要替换的样本数，最多替换 5 个
                    num_samples_to_replace = min(self.args['batch_size']//2 , (num_zeros - num_ones ) // 2)

                    ###从 document_representations_label_1，document_sequence_lengths_label_1 等中随机选择样本替换
                    for i in range(num_samples_to_replace):
                        replace_index = random.choice(indices_to_replace)
                        indices_to_replace.remove(replace_index)  
                        # 随机选择一个样本替换
                        random_index_label_1 = random.randint(0, len(indices_label_1) - 1)
                        
                        # 替换数据
                        batch_document_tensors[replace_index] = document_representations_label_1[random_index_label_1]
                        batch_document_sequence_lengths[replace_index] = document_sequence_lengths_label_1[random_index_label_1]
                        batch_news_tensors[replace_index] = news_representations_label_1[random_index_label_1]
                        batch_news_sequence_lengths[replace_index] = news_sequence_lengths_label_1[random_index_label_1]
                        fin_tensors[replace_index] = torch.tensor(fin_train_label_1[random_index_label_1], dtype=torch.float32).to(device=self.args['device'])
                        batch_correct_output[replace_index] = correct_output_label_1[random_index_label_1]
                    # print('batch_correct_output',num_samples_to_replace, batch_correct_output)
                    
                    
                    num_zeros = torch.sum(batch_correct_output[:,1].view(-1) == 0).item()  # 统计0的数量
                    num_ones = torch.sum(batch_correct_output[:,1].view(-1)  == 1).item()  # 统计1的数量
                    _bala +=  num_ones / ( num_zeros + num_ones)



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
            # plot each 2 labels
            # all_hidden_out_tensor = all_hidden_out
            # all_labels_tensor = all_labels
            # print('all_hidden_out_tensor', all_hidden_out_tensor.shape)
            # print('all_labels_tensor', all_labels_tensor.shape)

            # tsne = TSNE(n_components=2, random_state=42)

            # # 使用 t-SNE 进行降维
            # embedded = tsne.fit_transform(all_hidden_out_tensor.detach().cpu().numpy())

            # # 提取标签为 0 和 1 的样本的 t-SNE 降维结果和对应标签
            # embedded_label_0 = embedded[all_labels_tensor.detach().cpu().numpy() == 0]
            # embedded_label_1 = embedded[all_labels_tensor.detach().cpu().numpy() == 1]

            # # 绘制 t-SNE 图
            # plt.figure(figsize=(12, 6))
            # plt.scatter(embedded_label_0[:, 0], embedded_label_0[:, 1], label='Label 0', c='blue')
            # plt.scatter(embedded_label_1[:, 0], embedded_label_1[:, 1], label='Label 1', c='red')
            # plt.xlabel('t-SNE Dimension 1')
            # plt.ylabel('t-SNE Dimension 2')
            # plt.legend()
            # plt.title('t-SNE Visualization with Labels')
            # _path = os.path.join(self.args['model_directory'], "plot%s" % epoch)
            # plt.savefig(_path + '_TRAIN_tsne.png')
            # plt.close()
        ####################################################
            # hist plot
            # _path = os.path.join(self.args['model_directory'], "plot%s" % epoch)

            # self._dist(self.model.all_texts, _path+"all_texts")
            # self._dist(self.model.all_fin, _path+"all_fin")
            # self._dist(self.model.all_news, _path+"all_news")


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





















