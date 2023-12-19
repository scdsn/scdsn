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

from .functions import  *

from sklearn.metrics import accuracy_score, matthews_corrcoef
from .SupCon import SupConLoss
# from .class_aware_sampler import ClassAwareSampler
# from .MyCustomDataset import MyCustomDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import random

from matplotlib.colors import ListedColormap

# 假设 hidden_out_tsne 是 t-SNE 的降维结果，n1_color 是颜色标记列表

# 自定义颜色映射
colors = ['black', 'red', '#8B0000', '#800000']  # 分别对应 0、1、2、3

# 创建自定义颜色映射
cmap_custom = ListedColormap(colors)




random.seed(144000) 

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


utt_t_orig = None
utt_v_orig = None
utt_a_orig = None
utt_private_t = None
utt_private_v = None
utt_private_a = None
utt_shared_t = None
utt_shared_v = None
utt_shared_a = None

shared_or_private_p_t = None
shared_or_private_p_v= None
shared_or_private_p_a = None
shared_or_private_s = None
utt_t = None
utt_v= None
utt_a = None

utt_t_recon = None
utt_v_recon = None
utt_a_recon = None


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



class CustomAttentionLayer(nn.Module):
    def __init__(self, dim):
        super(CustomAttentionLayer, self).__init__()
        self.dim = dim

    def forward(self, tensor1, tensor2):
        # 计算 self-attention
        attention_scores = torch.bmm(tensor1.unsqueeze(1), tensor2.unsqueeze(2)).squeeze()
        attention_scores_scaled = attention_scores / (self.dim ** 0.5)
        print('attention_scores_scaled', attention_scores_scaled.shape)
        softmax_attention = torch.nn.functional.softmax(attention_scores_scaled, dim=1)
        weighted_tensor1 = torch.matmul(softmax_attention, tensor1)
        weighted_tensor2 = torch.matmul(softmax_attention.transpose(0, 1), tensor2)

        # 最终合并的结果
        combined_tensor = weighted_tensor1 + weighted_tensor2
        return combined_tensor

class MaxPoolingLayer(nn.Module):
    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, tensor1, tensor2):
        # 将两个张量按行连接（合并为一个张量）
        combined_tensor = torch.cat((tensor1, tensor2), dim=0)

        # 使用 Max Pooling 对合并后的张量进行池化操作
        pooled_tensor, _ = torch.max(combined_tensor, dim=0)
        
        return pooled_tensor

class CombineThreeMISASup(nn.Module):
    def __init__(self,args=None, n1=None, n2=None, n3=None):
        super(CombineThreeMISASup, self).__init__()

        if args is not None:
            self.args = vars(args)
        # self.n1 = torch.tensor(n1)
        self.n1 = n1

        assert self.args['labels'] is not None, "Must specify all labels in prediction"
        self.log = logging.getLogger()
        HIDDEN_SIZE = self.args['hid_fin']
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=2)
        self.onlytransformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer2 = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=2)
        self.dsntransformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=1)


        self.fusion = nn.Sequential() # only fusion
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        print('&&&&&&&&&&&&&&& dropout_rate', self.dropout_rate)
        if self.use_sigmoid==1:
            self.fusion.add_module('fusion_layer_2_', nn.Linear(self.args['hid_fin']+self.args['hid_trans']
                                                                , 1))
            self.fusion.add_module('fusion_layer_3', nn.Sigmoid())  
        else:
        # temp = bert_model_config.hidden_size // 2
            self.fusion.add_module('fusion_layer_2_', nn.Linear(self.args['hid_fin']+self.args['hid_trans']
                                                                ,2))
            self.fusion.add_module('fusion_layer_3', nn.Tanh())
        
        ## transfomrer
        self.mdafc = nn.Sequential(
                    nn.Linear(768, HIDDEN_SIZE),
                    nn.ReLU()
                )
        self.newsfc = nn.Sequential(
                    nn.Linear(768, HIDDEN_SIZE),
                    nn.ReLU()
                )

        ## 加上 MISA ############################################
        self.activation = nn.ReLU()
##
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.shared.add_module('shared_activation_1',self.activation)
        # self.shared.add_module('shared_2', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        # self.shared.add_module('shared_activation_2', nn.Sigmoid()  )
        # self.shared.add_msodule('s_norm', nn.LayerNorm(HIDDEN_SIZE,))


        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.private_t.add_module('private_t_activation_1', self.activation)
        # self.private_t.add_module('p_t_norm', nn.LayerNorm(HIDDEN_SIZE,))

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.private_v.add_module('private_v_activation_1', self.activation)
        # self.private_v.add_module('p_v_norm', nn.LayerNorm(HIDDEN_SIZE,))

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.private_a.add_module('private_a_activation_3', self.activation )
        # self.private_a.add_module('p_a_norm', nn.LayerNorm(HIDDEN_SIZE,))

##
        # Project to space ############ 可以自己改掉： 加一层layer norm
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE,))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(HIDDEN_SIZE,))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE,))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(HIDDEN_SIZE,))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE,))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(HIDDEN_SIZE))

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))


        self.transfc = nn.Sequential(
                    nn.Linear(HIDDEN_SIZE*6, self.args['hid_trans']),
                    nn.ReLU()
                )
        
        self.finfc = nn.Sequential(
                    nn.Linear(self.args['hid_fin'], HIDDEN_SIZE),
                    nn.ReLU()
                )
        ###############################################################

        # self.atten = CustomAttentionLayer(HIDDEN_SIZE)
        # self.maxpool = MaxPoolingLayer()
        # 检查包含指定关键字的参数名称
        spec = ['share', 'private', 'project', 'recon', 'trans']

        special_params = [param for name, param in self.named_parameters() if any(word in name for word in spec)]

        # 保留不包含指定关键字的参数名称
        norm_params = [param for name, param in self.named_parameters() if not any(word in name for word in spec)]

        # 设置不同参数组的学习率
        learning_rate_special = self.args['learning_rate'] * 1  # 设定特殊参数组的学习率
        learning_rate_normal = self.args['learning_rate'] * 1  # 设定普通参数组的学习率

        # 定义参数和对应的学习率
        params = [
            {"params": special_params, "lr": learning_rate_special},
            {"params": norm_params, "lr": learning_rate_normal},
        ]

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            params,
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )

        # self.optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     weight_decay=self.args['weight_decay'],
        #     lr=self.args['learning_rate']
        # )
        for name, param in self.named_parameters():

            if 'finextr' in name: 
                # print('finext')
                param.requires_grad = False
            if 'lstm' in name: 
                param.requires_grad = False
            param.requires_grad = False
            # if 'finextr' in name: 
            #     param.requires_grad = True
            if 'share' in name \
                or 'private' in name \
                or 'project' in name \
                or 'recon' in name:
                 param.requires_grad = True

            if 'transfc' in name\
            or  'newsfc' in name\
                or 'mdafc' in name\
                    or 'finfc' in name:
                 param.requires_grad = False

            if 'finfc' in name: # newsfc, mdafc, finfc
                 param.requires_grad = True

            if 'mdafc' in name: # newsfc, mdafc, finfc
                 param.requires_grad = True
            if 'newsfc' in name: # newsfc, mdafc, finfc
                 param.requires_grad = True
            if 'transfc' in name: # newsfc, mdafc, finfc
                 param.requires_grad = True
            if 'transform' in name:
                 param.requires_grad = True
            if 'fusion' in name:
                 param.requires_grad = True
            
            if (param.requires_grad):
                print(f"Parameter name: {name}")

        ##############################
        # DiffLoss
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.cmd_K = self.args['cmd_K']
        if self.args['use_CMD']==1:
            self.loss_cmd = CMD()
        else:
            self.loss_cmd = MMD_loss()
        ################################

        self.utt_t_orig  = utt_t_orig 
        self.utt_v_orig = utt_v_orig 
        self.utt_a_orig = utt_a_orig  

        self.utt_private_t = utt_private_t 
        self.utt_private_v  = utt_private_v 
        self.utt_private_a = utt_private_a
        self.utt_shared_t = utt_shared_t 
        self.utt_shared_v = utt_shared_v 
        self.utt_shared_a = utt_shared_a

        self.utt_t_recon = utt_t_recon 
        self.utt_v_recon = utt_v_recon 
        self.utt_a_recon = utt_a_recon

        ################################

        self.to("cuda:0")

    def forward(self,document_batch: torch.Tensor, 
                document_sequence_lengths: list, 
                device='cuda:0', fin=None, 
                news_batch=None, news_sequence_lengths=None):
        last_layer = self.mda(document_batch, document_sequence_lengths, device)
        
        last_layer = self.mdafc(last_layer)
        self.last_layer = last_layer = F.normalize(last_layer, dim=1)


        last_fin = self.fin(None, None, self.args['device'], fin, None, None)
        self.last_fin= last_fin = F.normalize(last_fin, dim=1)

        last_layer_news = self.news(news_batch, news_sequence_lengths, device)
        last_layer_news = self.newsfc(last_layer_news)
        self.last_layer_news = last_layer_news = F.normalize(last_layer_news, dim=1)

        last_trans_fin = self.finfc(last_fin)

        self.shared_private(last_layer, last_trans_fin, last_layer_news)

        h = torch.stack(( self.utt_private_t,  self.utt_private_v,  
                          self.utt_private_a, 
                         self.utt_shared_t, self.utt_shared_v,
                           self.utt_shared_a
                           ), dim=0)
        h = self.dsntransformer_encoder2(h)
        hinput_dsn = torch.cat((h[0], h[1], h[2]
                                , h[3], h[4], h[5]
                                ), dim=1)     
        
        prediction = self.fusion(hinput_dsn)

        return prediction


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

      
        ## reconstruct
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def get_diff_loss(self):

        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)#, _label)
        loss += self.loss_diff(private_v, shared_v)#, _label)
        loss += self.loss_diff(private_a, shared_a)# , _label)

        # Across privates
        # loss += self.loss_diff(private_a, private_t)#, _label)
        # loss += self.loss_diff(private_a, private_v)# , _label)
        # loss += self.loss_diff(private_t, private_v)#, _label)

        return loss
    
    def get_recon_loss(self):

        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)#, _label)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)#, _label)
        loss += self.loss_recon(self.utt_a_recon, self.utt_a_orig)#,  _label)
        loss = loss/3.0
        return loss

    def get_cmd_loss(self,):
        # losses between shared states
        # print('cmd K', self.cmd_K)
        loss = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, self.cmd_K)#, _label)
        loss += self.loss_cmd(self.utt_shared_t, self.utt_shared_a, self.cmd_K)#, _label)
        loss += self.loss_cmd(self.utt_shared_a, self.utt_shared_v, self.cmd_K)#, _label)
        loss = loss/3.0

        return loss
    def _dist(self, _INPUT, _path):
        batch_size = _INPUT.shape[0]
        dim = _INPUT.shape[1]

        # 创建一个空的张量来存储结果
        _ans = torch.empty((0, dim)).to("cuda:0")

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                a = _INPUT[i, :] - _INPUT[j, :]
                _ans = torch.cat((_ans, a.unsqueeze(0)), dim=0)
        
        # 检查是否有数据存在
        if _ans.numel() > 0:
            tensor_1d = _ans.view(-1)

            # 将张量转换为 NumPy 数组并绘制直方图
            data = tensor_1d.detach().cpu().numpy()

            plt.hist(data, bins=20)  # 设置 bins 数量
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of 2D Tensor Data')
            plt.savefig(_path + 'last_layer_difference_HIST.png')
            plt.close()
        else:
            print("No data to plot histogram.")


    def contrast(self,batch,_path, tuo=0.0001):# _neg_samples, label):
       
        last_layer = self.last_layer.clone()
        last_fin = self.last_fin.clone()
        last_layer_news = self.last_layer_news.clone()

    
       
        trasform1_last_layer = self._transform(last_layer, tuo)
        trasform1_last_fin = self._transform(last_fin,tuo)
        trasform1_last_layer_news = self._transform(last_layer_news, tuo)

        trasform1_last_layer = F.normalize(trasform1_last_layer, dim=1)
        trasform1_last_fin = F.normalize(trasform1_last_fin, dim=1)
        trasform1_last_layer_news = F.normalize(trasform1_last_layer_news, dim=1)

    

        # trans = [trans1, tarns2]
        # 2 fit 
        self.shared_private(
            trasform1_last_layer,
           trasform1_last_fin,
           trasform1_last_layer_news 
    )
        # # # print('self.utt_private_t', self.utt_private_t)
        h = torch.stack(( self.utt_private_t,  self.utt_private_v,  
                         self.utt_private_a, 
                         self.utt_shared_t, self.utt_shared_v,
                           self.utt_shared_a), dim=0)
        h = self.dsntransformer_encoder2(h)
        h = torch.cat((h[0], h[1], h[2],h[3],h[4],h[5]), dim=1)
        
   
        return h 


    def _transform(self, original_data, tuo):
        # Define your data augmentation parameters
        noise_std = tuo  # Noise coefficient
        scale_std = 0.05  # Scale coefficient
        mask_prob = 0.01  # Probability for mask
        batch_size = len(original_data)  # Your batch size
        feature_dim = original_data.shape[1]  # Dimensionality of your data

        # Define the sequence of data augmentation methods
        augmentation_sequence = ['noise' ]

        # Initialize the transformed data
        transformed_data = torch.zeros_like(original_data)
        device = "cuda:0"
        
        for i in range(len(original_data)):
            current_data = original_data[i].clone()  # Clone the original data

            # Apply transformations with 0.5 probability in the specified sequence
            # for transformation in augmentation_sequence:
            if  random.random() < 0.9:
                noise = torch.randn(1, feature_dim).to(device) * noise_std
                # self._dist(noise)
                current_data = current_data + noise
                # elif transformation == 'scale' and random.random() < 0.5:
                #     scale_factor = torch.randn(1).to(device) * scale_std + 1
                #     current_data = current_data * scale_factor
                # elif transformation == 'mask' and random.random() < 0.5:
                #     mask = (torch.rand(1, feature_dim).to(device) > mask_prob).float()
                #     current_data = current_data * mask
                # elif transformation == 'translation' and random.random() < 0.5:
                #     start_point = torch.randint(0, feature_dim, (1,)).to(device)
                #     shifted_data = torch.cat((current_data[start_point[0]:], current_data[:start_point[0]]))
                #     current_data = shifted_data

            transformed_data[i] = current_data

        return transformed_data

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

        if self.args['use_sigmoid']==1:
            loss_weight = (1.0 / ( correct_output[:,1].sum() / correct_output.shape[0]) ).to(device=self.args['device'])

        else:
            loss_weight = ((correct_output.shape[0] / torch.sum(correct_output, dim=0))-1).to(device=self.args['device'])
     
        if self.args['sampler'] == 0:
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        else:
            # print("&&&&&&&&&&&&&&&&&&&   no weight")
            self.loss_function = torch.nn.BCEWithLogitsLoss()#pos_weight=loss_weight)
        # self.loss_function = CustomBCEWithPosWeightLoss(pos_weight=loss_weight)

        self.loss_function_con = SupConLoss(device = "cuda:0",
                                            temperature=self.args['temp'])




        pretrained_dict = torch.load(self.args['fin_path'])
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if 'finextr' in key}

        self.load_state_dict(pretrained_dict, strict=False)
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
        _all_s_list = []
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

            all_hidden_out = torch.empty(0, 129).to(self.args['device'])  #
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
                n1_b = self.n1[i:i + self.args['batch_size']]

                batch_predictions = self.forward(batch_document_tensors,
                                                batch_document_sequence_lengths, self.args['device'],
                                                fin_tensors, 
                                                batch_news_tensors, 
                                                batch_news_sequence_lengths)



                if self.args['use_sigmoid']==1:
                    # print('batch_correct_output', batch_predictions.view(-1).shape, batch_correct_output)
                    # print('batch_predictions', batch_correct_output[:,1].view(-1).shape,  batch_predictions)

                    loss = self.loss_function(batch_predictions.view(-1),
                                            batch_correct_output[:,1].view(-1))
                else:
                    loss = self.loss_function(batch_predictions,
                                            batch_correct_output)
                cls_ += loss
                if self.args['only_fusion'] != 1:
                    cmd_loss = self.get_cmd_loss()
                    diff_loss = self.get_diff_loss()
                    recon_loss = self.get_recon_loss()
                    loss += (cmd_loss * self.args['cmd_W'] + 
                             diff_loss * self.args['diff_W']
                             + recon_loss * self.args['recon_W'] )

                    diff_ += diff_loss
                    cmd_ += cmd_loss
                    recon_ += recon_loss


                _label = batch_correct_output[:,1].view(-1)

                if self.args['contrast']==1:
                    # print(self.args['contrast'])
                    
                    _labels = batch_correct_output[:,1].view(-1)
                    _path = os.path.join(self.args['model_directory'], "plot%s" % epoch)

                    hidden_out = self.contrast(n_, _path, self.args['tuo'])#_neg_samples, _labels)

                    ## collection

                    # _labels = torch.cat([_labels, n1_], dim=0).view(-1)
                    hidden_out = F.normalize(hidden_out, dim=1)
                    # print(hidden_out.shape)
                    all_hidden_out = torch.cat((all_hidden_out, hidden_out), dim=0)

                   
                    # plt.savefig(_path)

                    # self.utt_shared_t = F.normalize(self.utt_shared_t, dim=1)
                    # self.utt_shared_v = F.normalize(self.utt_shared_v, dim=1)
                    # self.utt_shared_a = F.normalize(self.utt_shared_a, dim=1)
                    # self.utt_private_t = F.normalize(self.utt_private_t, dim=1)
                    # self.utt_private_v = F.normalize(self.utt_private_v, dim=1)
                    # self.utt_private_a = F.normalize(self.utt_private_a, dim=1)

                    
             
                    if self.args['contrast_mid'] == 1:
                        pass
                        # cont_s = cont_loss_s1 + cont_loss_s2 + cont_loss_s3
                        # cont_p =  cont_loss_p1 + cont_loss_p2 + cont_loss_p3
                    else:
                        cont_s = self.loss_function_con(hidden_out, 
                    _labels)
                    
                
                    cont_s = cont_s * 1
                   

                    all_s += cont_s
                 
                    # all_p += cont_p
                    # #### END ########################
                    # loss +=  all_s / 

                n_ += 1
                
                epoch_loss += float(loss.item())
                
                #self.log.info(batch_predictions)
                loss.backward()
                torch.nn.utils.clip_grad_value_([param for param in self.parameters() if param.requires_grad], 1)
                # torch.nn.utils.clip_grad_norm_([param for param in self.parameters() if param.requires_grad], max_norm=0.5)

                self.optimizer.step()
                self.optimizer.zero_grad()

                
            ## in epoch for

            epoch_loss /= document_representations.shape[0] / self.args['batch_size']  # divide by number of batches per epoch
            print("=============== TRAIN ================== ")
          
          
       
            self.log.info('bala: %f', _bala / n_)
            self.log.info('cls: %f', cls_ / n_)
            self.log.info('diff: %f', diff_ / n_)
            self.log.info('sim: %f', cmd_ / n_)
            self.log.info('recon: %f', recon_ / n_)
            _all_cls_loss.append(cls_ / n_)
            
            _all_sim_loss.append(cmd_ / n_)
            _all_diff_loss.append(diff_ / n_)
            _all_recon_loss.append(recon_ / n_)

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
                _all_s_list.append(all_s / n_)
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
                # self.tse_vis(_path,)# _labels)
            ## each epoch

            ## plot
                _path = os.path.join(self.args['model_directory'], "plot%s" % epoch)
                _PLOT = 1# Plot individual loss graphs
                
                # print('_all_diff_loss', _all_diff_loss)
                if _PLOT:
                    plt.figure(figsize=(12, 6))
                    plt.plot(range(1,_all_epoch+1), torch.tensor(_all_s_list).detach().cpu().numpy(), label='Train')
      
                    plt.xlabel('Epoch')
                    plt.ylabel('LOSS')
                    plt.legend()
                    plt.savefig(_path + 'ALL_SHARED_LOSS.png')
                    plt.close()

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

##############################

            correct_output = torch.FloatTensor(data[1]).transpose(0,1)
            assert self.args['labels'] is not None

        self.eval()
        # print('correct_output', correct_output)
        with torch.no_grad():
            if self.args['use_sigmoid']==1:
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
                if self.args['use_sigmoid']==1:

                    loss = self.loss_function(prediction.view(-1),
                                           batch_correct_output[:,1].view(-1))
                else:
                    loss = self.loss_function(prediction,
                                           batch_correct_output)
                cls_ += loss
                n_ += 1
                # print('batch_correct_output', batch_correct_output.shape)
                _label = batch_correct_output[:,1].view(-1)
                if self.args['only_fusion'] != 1:
                    
                    cmd_loss = self.get_cmd_loss()
                    diff_loss = self.get_diff_loss()
                    recon_loss = self.get_recon_loss()

                    cmd_ += cmd_loss
                    diff_+= diff_loss
                    recon_ += recon_loss
                    # collect v6s
                    _labels = [i for i in batch_correct_output[:,1]]
          

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
   
       

        if self.args['use_sigmoid']==1:
            _sig = scores = predictions
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

            roc_auc = roc_auc_score(correct_output[1,:].view(-1), _sig)
            
      
        
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

        torch.save(self.state_dict(), os.path.join(checkpoint_path, 'net1.pkl')  )

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

        # net = self.document_bertlstm_news
        # if isinstance(self.document_bertlstm_news, nn.DataParallel):
        #     net = self.document_bertlstm_news.module
        # torch.save(net.state_dict(), os.path.join(checkpoint_pathnews, WEIGHTS_NAME))
        # #save configurations
        # net.config.to_json_file(os.path.join(checkpoint_pathnews, CONFIG_NAME))
        # #save exact vocabulary utilized
        # self.bert_tokenizer.save_vocabulary(checkpoint_pathnews)





















