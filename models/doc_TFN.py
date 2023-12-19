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


class SubNet(nn.Module):
    '''    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # print()
        self.in_size = in_size
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # print("insize", self.in_size)
        # print("x", x.shape, x)

        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        #self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        # _, final_states = self.rnn(x)
        h = self.dropout(x)
        y_1 = self.linear_1(h)
        return y_1


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self,args,input_dims=(768, 768, 768), hidden_dims=(128,128,128), 
                 text_out=128, 
                dropouts=(0.15, 0.15, 0.15, 0.15), post_fusion_dim=128):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out= text_out
        self.post_fusion_dim = post_fusion_dim

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


        if args is not None:
            self.args = vars(args)
        print('labels',self.args['labels'])
        assert self.args['labels'] is not None, "Must specify all labels in prediction"

        self.log = logging.getLogger()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])
        self.bertnews_tokenizer = BertTokenizer.from_pretrained(self.args['newsbert_model_path'])


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

        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))

         #account for some random tensorflow naming scheme
        if os.path.exists(self.args['newsbert_model_path']):
            if os.path.exists(os.path.join(self.args['newsbert_model_path'], CONFIG_NAME)):
                news_config = BertConfig.from_json_file(os.path.join(self.args['newsbert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['newsbert_model_path'], 'bert_config.json')):
                news_config = BertConfig.from_json_file(os.path.join(self.args['newsbert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            news_config = BertConfig.from_pretrained(self.args['newsbert_model_path'])
        news_config.__setattr__('num_labels',len(self.args['labels']))
        news_config.__setattr__('bert_batch_size',self.args['bert_batch_size'])

        self.bert_doc_classification = DocumentBertLSTMNews.from_pretrained(self.args['bert_model_path'], config=config, 
                                    # only_fusion=self.args['only_fusion'],
                                    # use_sigmoid=self.args['use_sigmoid']
                                    )
        self.bert_doc_classification.freeze_bert_encoder()
        self.bert_doc_classification.unfreeze_bert_encoder_last_layers()
        self.document_bertlstm_news = DocumentBertLSTMNews.from_pretrained(self.args['newsbert_model_path'], config=news_config, 
                            # only_fusion=self.args['only_fusion'],
                            # use_sigmoid=self.args['use_sigmoid']
                            )
        self.document_bertlstm_news.freeze_bert_encoder()
        self.document_bertlstm_news.unfreeze_bert_encoder_last_layers()
        
        self.bert_batch_size = self.bert_doc_classification.bert_batch_size 
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

        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self._W = _W = self.args['_W']
        # self.loss_diff = DiffLoss(_W)
        # self.loss_recon = MSE(_W)
        # self.loss_cmd = CMD(_W)
        # if _USE_WEIGHT:
        self.loss_diff = DiffLoss(_W)
        self.loss_recon = MSE(_W)
        self.loss_cmd = CMD(_W)
        # else:
        # self.loss_diff = DiffLoss()
        # self.loss_recon = MSE()
        # self.loss_cmd = CMD()

        # self.model = self.bert_doc_classification
        # self.loss_function_con = SupConLoss()
        self.fc1 = nn.Linear(418, 768)  # 隐藏层
        self.to("cuda:0")


    def forward(self,  document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda', fin=None, news_batch=None, news_sequence_lengths=None):
        
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_doc_classification.bert_batch_size),
                                              self.bert_doc_classification.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.bert_doc_classification.dropout(self.bert_doc_classification.bert(document_batch[doc_id][:self.bert_doc_classification.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_doc_classification.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_doc_classification.bert_batch_size,2])[1])

        output, (_, _) = self.bert_doc_classification.lstm(bert_output.permute(1,0,2))

        self.last_layer = last_layer = output[-1]

        bert_news_output = torch.zeros(size=(news_batch.shape[0],
                                              min(news_batch.shape[1],self.document_bertlstm_news.bert_batch_size),
                                              self.document_bertlstm_news.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(news_batch.shape[0]):
            bert_news_output[doc_id][:self.bert_batch_size] = self.document_bertlstm_news.dropout(self.document_bertlstm_news.bert(news_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=news_batch[doc_id][:self.document_bertlstm_news.bert_batch_size,1],
                                            attention_mask=news_batch[doc_id][:self.document_bertlstm_news.bert_batch_size,2])[1])

        output_news, (_, _) = self.document_bertlstm_news.lstm(bert_news_output.permute(1,0,2))

        self.last_layer_news = last_layer_news = output_news[-1]
    
# ##      
        # print(self.bert_model_config.hidden_size)

        self.last_fin = last_fin = self.fc1(fin)
        # print('last_layer', last_layer) # 3* 768
        self.last_layer = F.normalize(self.last_layer, dim=1)
        self.last_fin = F.normalize(self.last_fin, dim=1)
        self.last_layer_news = F.normalize(self.last_layer_news, dim=1)
        
        return self._fusion(self.last_layer, self.last_fin, self.last_layer_news)
    


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

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3 #* self.output_range + self.output_shift

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
        del train
        del dev
        del test

        self.train()
        self.document_bertlstm_news.train()
        self.bert_doc_classification.train()

        print(" encode start", )

        document_representations, document_sequence_lengths  = encode_documents(train_documents, self.bert_tokenizer)
        news_representations, news_sequence_lengths  = encode_documents(news_train, self.bertnews_tokenizer)

        correct_output = torch.FloatTensor(train_labels)
        # print(correct_output)
        loss_weight = (1.0 / ( correct_output[:,1].sum() / correct_output.shape[0]) ).to(device=self.args['device'])
        # print('loss_weight', loss_weight)
        # loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1).to(device=self.args['device'])

        self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        # self.loss_function = CustomBCEWithPosWeightLoss(pos_weight=loss_weight)

        assert document_representations.shape[0] == correct_output.shape[0]
        assert news_representations.shape[0] == correct_output.shape[0]

        if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])
        self.document_bertlstm_news.to(device=self.args['device'])
        
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
            # 获取数据集的大小
            data_size = document_representations.shape[0]

            # 原地洗牌数据
            for i in range(data_size - 1, 0, -1):
                j = torch.randint(0, i + 1, (1,)).item()
                document_representations[i], document_representations[j] = document_representations[j], document_representations[i]
                document_sequence_lengths[i], document_sequence_lengths[j] = document_sequence_lengths[j], document_sequence_lengths[i]
                correct_output[i], correct_output[j] = correct_output[j], correct_output[i]
            
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
           

            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                self.optimizer.zero_grad()
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_document_sequence_lengths= document_sequence_lengths[i:i+self.args['batch_size']]
                #self.log.info(batch_document_tensors.shape)
                
                batch_news_tensors = news_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_news_sequence_lengths= news_sequence_lengths[i:i+self.args['batch_size']]
              

                fin_tensors = torch.tensor(fin_train[i:i + self.args['batch_size']], dtype=torch.float32).to(device=self.args['device'])
        
        
                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                

                batch_predictions = self.forward(batch_document_tensors,
                                                batch_document_sequence_lengths, self.args['device'],
                                                fin_tensors, 
                                                batch_news_tensors, 
                                                batch_news_sequence_lengths)

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
         
    
            epoch_loss /= document_representations.shape[0] / self.args['batch_size']  # divide by number of batches per epoch
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
                _PLOT = 0# Plot individual loss graphs
                
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
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss).detach().cpu().numpy(), label='SIM Train Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss_val).detach().cpu().numpy(), label='SIM Val Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_sim_loss_test).detach().cpu().numpy(), label='SIM Test Loss')

                    plt.xlabel('Epoch')
                    plt.ylabel('SIM Loss')
                    plt.legend()

                    plt.subplot(2, 2, 3)
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss).detach().cpu().numpy(), label='DIFF Train Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss_val).detach().cpu().numpy(), label='DIFF Val Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_diff_loss_test).detach().cpu().numpy(), label='DIFF Test Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('DIFF Loss')
                    plt.legend()

                    plt.subplot(2, 2, 4)
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss).detach().cpu().numpy(), label='RECON Train Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss_val).detach().cpu().numpy(), label='RECON Val Loss')
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_recon_loss_test).detach().cpu().numpy(), label='RECON Test Loss')
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


        if isinstance(data, list):
            document_representations, document_sequence_lengths = encode_documents(data, self.bert_tokenizer)
            news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)

        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            document_representations, document_sequence_lengths = encode_documents(data[0], self.bert_tokenizer)
            news_representations, news_sequence_lengths = encode_documents(news_dev, self.bertnews_tokenizer)

            correct_output = torch.FloatTensor(data[1]).transpose(0,1)
            assert self.args['labels'] is not None
        self.eval()
        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        self.document_bertlstm_news.to(device=self.args['device'])
        self.document_bertlstm_news.eval()
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
            ## 
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

        scores = predictions[:,1].view(-1)
        # print('scores', scores.shape, scores)
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
        
    
        assert correct_output.shape == predictions.shape
        precisions = []
        recalls = []
        fmeasures = []

        for label_idx in range(predictions.shape[0]):

            correct = correct_output[label_idx].cpu().view(-1).numpy()
            predicted = predictions[label_idx].cpu().view(-1).numpy()
            print('correct', correct)
            print('predicted', predicted)

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

        self.train()
        self.bert_doc_classification.train()
        self.document_bertlstm_news.train()
        
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

        #save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        #save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        #save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)

        if not os.path.exists(checkpoint_pathnews):
            os.mkdir(checkpoint_pathnews)
        else:
            raise ValueError("Attempting to save checkpoint to an existing directory")
        self.log.info("Saving checkpoint: %s" % checkpoint_pathnews )

        net = self.document_bertlstm_news
        if isinstance(self.document_bertlstm_news, nn.DataParallel):
            net = self.document_bertlstm_news.module
        torch.save(net.state_dict(), os.path.join(checkpoint_pathnews, WEIGHTS_NAME))
        #save configurations
        net.config.to_json_file(os.path.join(checkpoint_pathnews, CONFIG_NAME))
        #save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_pathnews)



