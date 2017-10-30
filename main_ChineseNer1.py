# encoding=utf8
#'''
#训练命令:python main.py --train=True --clean=True;训练过程大概耗时3个小时
#测试命令:python main.py 
#'''
import os
import pickle
import itertools
#OrderedDict实现了字典元素的顺序存储
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from blstm_ner import BLSTM_Model
from loader import load_sentences, update_tag_scheme,char_mapping, tag_mapping
from loader import augment_with_pretrained
from data_utils import get_logger, make_path, clean
from data_utils import print_config, save_config, load_config, test_ner
from data_utils import input_from_line, BatchManager,prepare_dataset

flags = tf.app.flags
#当以下两项为False时，处于测试状态;当以下两项为True时，处于训练状态;
flags.DEFINE_string('mode', 'demo', 'train/test/demo')
flags.DEFINE_boolean("clean",      False,      "clean train folder")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
#type模式
#I表示实体的中间，O表示非实体，B表示实体的开始，E表示实体的结束，S表示独立的实体
flags.DEFINE_string("tag_type",   "iobes",    "tagging type iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_integer("num_epoches",    50,         "num of epoches")
flags.DEFINE_integer("batch_size",    50,         "batch size")
flags.DEFINE_float("learning_rate", 0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Whether use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Whether replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Whether lower case")

flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_dir",    ".\checkpoints",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "log/train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
#evaluation script 评估脚本
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "results",       "Path for evaluation results")
flags.DEFINE_string("emb_file",     "data/wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")
FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.learning_rate > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

# config for the model
def config_model(char_to_id, tag_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)  #求汉字(不算重复)的总个数
    config["char_dim"] = FLAGS.char_dim  #字向量的维数
    config["num_tags"] = len(tag_id)   #求命名实体的种类个数
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim      
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_file    #已训练好的词向量文件
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["learning_rate"] = FLAGS.learning_rate
    config["tag_type"] = FLAGS.tag_type
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    config["ckpt_dir"]=FLAGS.ckpt_dir
    return config

def Evaluate(sess, model, name, data, id_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_tag)
    #test_ner用于计算结果的准确率，召回率等
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))


 # load data sets
train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
#train_sentences为三维列表，每一句话构成一个列表，而每一句话的元素为汉字对应的命名实体类型[厦,B-LOC]
dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, FLAGS.tag_type)
update_tag_scheme(dev_sentences, FLAGS.tag_type)
update_tag_scheme(test_sentences, FLAGS.tag_type)

# create maps if not exist
if not os.path.isfile(FLAGS.map_file):
    # create dictionary for word
    if FLAGS.pre_emb:
        #dict_chars_train为一个字典，键为每个汉字，值为该汉字在这段语料库中出现的次数
        dict_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
        #将该语料库所有汉字按出现次数的多少进行排列
        #dict_chars为所有汉字组成的字典，包括训练集和测试集，值为该汉字在这段语料库中出现的次数
        #char_to_id为字典，键为出现次数从大到小排列的汉字，值为该汉字对应的序号，如'的':3 
        #id_to_char为字典，键为0-最后一个汉字，值为出现次数从大到小排列的汉字
        #{0: '<PAD>', 1: '<UNK>', 2: '，', 3: '的', 4: '。', 5: '国', 6: '一'
        dict_chars, char_to_id, id_to_char = augment_with_pretrained(
            dict_chars_train.copy(),FLAGS.emb_file,
            list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])))
    else:
        _, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

    # Create a dictionary and a mapping for tags
    #tag_id为字典，键为NER状态，值为序号;id_tag为字典，键为序号，值为NER状态
    _t, tag_id, id_tag = tag_mapping(train_sentences)
    #tag_id:{'O': 0, 'I-ORG': 1, 'B-LOC': 2, 'E-LOC': 3, 'B-ORG': 4,
    #'E-ORG': 5, 'I-LOC': 6, 'I-PER': 7, 'B-PER': 8, 'E-PER': 9, 'S-LOC': 10, 
    #'S-PER': 11, 'S-ORG': 12}
    with open(FLAGS.map_file, "wb") as f:
        #将这些参数写入FLAGS.map_file中
        pickle.dump([char_to_id, id_to_char, tag_id, id_tag], f)
else:
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_id, id_tag = pickle.load(f)
print('共有%d个汉字' %len(char_to_id))
# prepare data, get a collection of list containing index
train_data = prepare_dataset(train_sentences, char_to_id, tag_id, FLAGS.lower)
#20864条数据
#train_data=[string, chars, segs, tags]
#string为一句话中的汉字构成的列表;chars为一句话中的汉字的序号构成的列表;
#segs为每一句话分词后的汉语词汇的编码(一个字组成的词编码为0;两个字组成的词为1,3;
#三个字组成的词为1,2,3);tags为一句话中该汉字对应的NER状态的序号
#train_data[0]=[['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与','金',
#'门', '之', '间', '的', '海', '域', '。'], [235, 1574, 153, 152, 30, 236, 8,
#1500, 238, 89, 182, 238, 112, 198, 3, 235, 658, 4], [1, 3, 1, 3, 1, 3, 0, 1, 
#3, 0, 1, 3, 1, 3, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 2, 3, 0, 0, 0,
#0, 0, 0]]
#2318条数据
dev_data = prepare_dataset(dev_sentences, char_to_id, tag_id, FLAGS.lower)
#4636条数据
test_data = prepare_dataset(test_sentences, char_to_id, tag_id, FLAGS.lower)
print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))
#train_manager为训练数据的集合
'''
train_manager为三维列表:
第一层为总的批次数;第二层为四个列表string,chars,segs,tags;
分别对应:每一批数据string组成的列表，每一批数据chars组成的列表
每一批数据segs组成的列表,每一批数据tags组成的列表
'''
train_manager = BatchManager(train_data, FLAGS.batch_size)
dev_manager = BatchManager(dev_data, FLAGS.batch_size)
test_manager=BatchManager(test_data, FLAGS.batch_size)
# make path for store log and model if not exist
make_path(FLAGS)
if os.path.isfile(FLAGS.config_file):
    config = load_config(FLAGS.config_file)
else:
    config = config_model(char_to_id, tag_id)
    save_config(config, FLAGS.config_file)
#此路径名为log/train.log
logger = get_logger(FLAGS.log_file)
#打印config信息，并写入/log/train.log中
print_config(config, logger)
# limit GPU memory
tf_config = tf.ConfigProto()
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会
#释放内存，所以会导致碎片
tf_config.gpu_options.allow_growth = True
steps_per_epoch = train_manager.num_batch   #表示将所有数据分成的批数
with tf.Session(config=tf_config) as sess:
     model =BLSTM_Model(config,logger)
     # saver of the model
     saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
     model.build_graph(sess,id_to_char)
     if FLAGS.mode=='train':
        if FLAGS.clean:
           clean(FLAGS)
        logger.info("start training")
        loss = []
        count=0
        #总循环50次
        best_accuracy=0
        for epoch in range(FLAGS.num_epoches):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                   iteration = step // steps_per_epoch + 1
                   logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                   loss = []
            #dev_manager为三维列表:
            #第一层为总的批次数;第二层为四个列表string,chars,segs,tags;
            #分别对应:每一批数据string组成的列表，每一批数据chars组成的列表
            #每一批数据segs组成的列表,每一批数据tags组成的列表
            Evaluate(sess, model, "dev", dev_manager, id_tag, logger)
            count+=1
            save_path=saver.save(sess, os.path.join(FLAGS.ckpt_dir,"ner.ckpt"), global_step=count)
     elif FLAGS.mode=='test':
          ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
          if ckpt: 
             saver.restore(sess, ckpt)
             logger.info("Reading model parameters from %s" % ckpt)
             Evaluate(sess, model, "test", test_manager, id_tag, logger)
     else:  
         try:
             ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
             if ckpt: 
                saver.restore(sess, ckpt)
                logger.info("Reading model parameters from %s" % ckpt)
             #测试句子
             line = "深圳是华为技术有限公司的总部，毛泽东是湖南湘潭人，邓小平曾任中共中央总书记，姚明曾在休斯顿火箭队打球。"
             test_line=input_from_line(line, char_to_id)
             print(test_line)
             result = model.evaluate_line(sess,test_line,id_tag)
             print(result)
         except Exception as e:
                logger.info(e)



