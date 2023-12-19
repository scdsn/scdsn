

import sys
new_path = r'C:\Users\FxxkDatabase\Desktop\bert_document_classification-master\bert_document_classification-master'
sys.path.append(new_path)

from bert_document_classification.prepare_data import load_data

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from bert_document_classification.document_bert import BertForDocumentClassification
from pprint import pformat
from bert_document_classification.modelTFN import TFN

from bert_document_classification.modelLR import LMF
from bert_document_classification.Fin import Fin
from bert_document_classification.doc_onetext import BertForDocumentClassification_ONLY
from bert_document_classification.doc_three import CombineThree
from bert_document_classification.doc_three_MISA import CombineThreeMISA
from bert_document_classification.doc_three_MISASupCon import CombineThreeMISASup
from bert_document_classification.base_selfMM import SelfMM
from bert_document_classification.doc_thtwo_HAN import *


import time, logging, torch, configargparse, os, socket
import pickle
import pandas as pd

log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)
    p.add('--newsbert_model_path', help='Model path to BERT')

    p.add('--contrast', help='con', type=int)
    p.add('--only_fusion', help='con', type=int)
    p.add('--use_sigmoid', help='con', type=int)
    p.add('--cmd_K', help='con', type=int)
    p.add('--tuo', help='con', type=float)
    p.add('--use_CMD', help='con', type=int)
    p.add('--sampler', help='con', type=int)
    p.add('--temp', help='con', type=float)
    p.add('--contrast_mid', help='con', type=int)
    p.add('--cmd_W', help='con', type=float)
    p.add('--diff_W', help='con', type=float)
    p.add('--recon_W', help='con', type=float)
    p.add('--dr', help='con', type=float)
    p.add('--encoding', help='con', type=int)
    p.add('--hid_fin', help='con', type=int)
    p.add('--mda', help='con', type=int)
    p.add('--hidden_dim', help='con', type=int)
    p.add('--fin_path', help='con',)

    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]
    print('freeze_bert', args.freeze_bert)



    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.dev = 'cpu'

    return args

PORT_ = 1.0 ## portion of data for debug purpose

if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=["config.ini"])
    args = _initialize_arguments(p)

    torch.cuda.empty_cache()

    # 1 read df
    # construct col1 col2
    # merged_df= pd.read_pickle('../../../../data/mdf1104_all.pkl')
    merged_df= pd.read_pickle('../../../../data/merge1214_all.pkl')

    print('[READ data]',merged_df.shape, merged_df.columns)
    merged_df['column1'] = merged_df['next_year_label'].apply(lambda x: 1 if x == 0 else 0)
    merged_df['column2'] = merged_df['next_year_label'].apply(lambda x: 0 if x == 0 else 1)

    # 2 split train dev test

    # train_data = merged_df[(merged_df['year'] > 2011) & (merged_df['year'] < 2018)]
    # val_data = merged_df[(merged_df['year']==2018 ) | (merged_df['year']==2019)]
    # test_data = merged_df[(merged_df['year']==2020 )  | (merged_df['year']==2021) ]
    # print('[SPLIT]', train_data.shape, val_data.shape, test_data.shape)


    train_data = merged_df[(merged_df['year'] >= 2011) & (merged_df['year'] < 2020)]
    val_data = merged_df[merged_df['year']==2020]
    test_data = merged_df[merged_df['year']==2021]
    print('[SPLIT]', train_data.shape, val_data.shape, test_data.shape)

    # 3 text, train, dev, test
    train_text = [i for i in train_data['operatingstatement'] ]
    train_label = train_data[['column1', 'column2' ]]
    train_label = train_label.values.tolist()

    val_text = [i for i in val_data['operatingstatement'] ]
    val_label = val_data[['column1', 'column2' ]]
    val_label = val_label.values.tolist()

    test_text = [i for i in test_data['operatingstatement'] ]
    test_label = test_data[['column1', 'column2' ]]
    test_label = test_label.values.tolist()

    #  4 fin train, dev, test
    #  
    with open("../../../../data/ALL_COLS.pkl", "rb") as file:
        COLS = pickle.load(file)
    # COLS = COLS[:-12]
    train_fin = train_data[COLS].values
    val_fin = val_data[COLS].values
    test_fin = test_data[COLS].values

    train_news = [i for i in train_data['texts'] ]
    val_news = [i for i in val_data['texts'] ]
    test_news = [i for i in test_data['texts'] ]
    

    print('[train_text shape]', len(train_text), len(val_text))
    print('[FIN shape]', train_fin.shape, val_fin.shape) 
    print('[train_news shape]', len(train_news), len(val_news))


    #documents and labels for training
    print('train_documents', train_text[0][:10])
    print('train_news', train_news[0][:10])
    n1 = train_data['ID'].values
    n2 = val_data['ID'].values
    n3 = test_data['ID'].values
    # model = BertForDocumentClassification(args=args)
    # model = TFN(args=args)
    # model = LMF(args=args)
    # model = Fin(args = args)
    # model = BertForDocumentClassification_ONLY(args=args)
    # model = CombineThree(args)
    # model = CombineThreeMISA(args)
    model = CombineThreeMISASup(args, n1,n2,n3)
    # model = SelfMM(args)
    
   
    print('START TRAINING...' )
    train_len = int(len(train_label) *PORT_)
    val_len = int(len(val_label) * PORT_) 
    test_len = int(len(test_label) *PORT_)
    print('train_len', train_len)
    print('val_len', val_len)

    model.fit((train_text[-train_len:], train_label[-train_len:]), 
              (val_text[:val_len], val_label[:val_len]), 
              train_fin[-train_len:], 
              val_fin[:val_len],
                train_news[-train_len:],
                  val_news[:val_len],
               (test_text[:test_len], test_label[:test_len]), 
               test_fin[:test_len], test_news[:test_len] )
    
   