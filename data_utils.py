# encoding = utf8
import math
import random
import numpy as np


import os
import re
import json
import shutil
import logging
from conlleval import return_report
import jieba
jieba.initialize()

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    #创建一个格式并把它添加到指定的handlers
    file_h = logging.FileHandler(log_file)
    #指定写入文件的最低分发log信息的级别，这里logging模块只会输出DEBUG及以上的log
    file_h.setLevel(logging.DEBUG)
    stream_h = logging.StreamHandler()
    #指定输出到屏幕的最低分发log信息的级别，这里logging模块只会输出INFO及以上的log
    stream_h.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_h.setFormatter(formatter)
    file_h.setFormatter(formatter)
    logger.addHandler(stream_h)
    logger.addHandler(file_h)
    return logger

def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:  #block为每一句话的数据
            for line in block: #line为每一句话中的每个单词的数据('纽 B-LOC B-LOC')
                to_write.append(line + "\n")
            to_write.append("\n")  #表示一句话结束
        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines

def print_config(config, logger):
    """
    Print configuration of the model
    打印模型的配置
    """
    for k, v in config.items():
        logger.info("{}: {}".format(k.ljust(15), v))

def make_path(params):
    """
    Make folders for training and evaluation
    """
    #判断给定的目录是否存在，若不存在，则创建该目录
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_dir):
        os.makedirs(params.ckpt_dir)
    if not os.path.isdir("log"):
        os.makedirs("log")

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        #将数据以json格式写入文件中
        json.dump(config, f, ensure_ascii=False, indent=4)
        
def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        #将数据从json格式文件中读出
        return json.load(f)

def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(open(emb_path, 'r', encoding='utf-8')):
        line = line.rstrip().split()
        #line为一个汉字+其汉字对应的100维向量，共101维
        #确保原embedding层的维数和现在网络的维数相同
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        #word为该序号对应的汉字
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights

def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)  #独立词编号为0
        else:
            #两个字构成的词为[1,3]
            #三个字构成的词为[1,2,3]
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature

def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]  #string为一句话中的汉字构成的列表
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        #chars为每一句话中的汉字的序号构成的列表
        #"".join(string)将列表转化为字符串
        segs = get_seg_features("".join(string))
        #segs为每一句话分词后的汉语词汇的编码
        if train:
            #tags为一句话中该汉字对应的NER状态的编号
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])
    return data

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob2iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def iobes2iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags



def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by 0 with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words

def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs

def full_to_half(s):
    """
    Convert full-width character to half-width one 
    将全角字符转换为半角字符
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    """
    Cut text to sentences 
    """
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut=True
            pre_cut=False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut=True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    #.replace(a,b)用b代替a
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")

    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    inputs.append([[len(line)]])  #该句话的句子长度
    return inputs


class BatchManager(object):
    '''
    data由下面四部分构成:
    string为一句话中的汉字构成的列表;chars为一句话中的汉字的序号构成的列表;
    segs为每一句话分词后的汉语词汇的编码;tags为一句话中该汉字对应的NER状态的序号
    '''
    def __init__(self, data,  batch_size):
        '''
        self.batch_data为三维列表:
        第一层为总的批次数;第二层为四个列表string,chars,segs,tags;
        分别对应:每一批数据string组成的列表，每一批数据chars组成的列表
        每一批数据segs组成的列表,每一批数据tags组成的列表
        '''
        self.batch_data = self.sort_and_pad(data, batch_size)
        #self.batch_data为每一batch的数据为元素组成的列表
#        self.len_data = len(self.batch_data)  #总共有self.len_data个batch
    def sort_and_pad(self, data, batch_size):
        #ceil向上取整
        self.num_batch = math.ceil(len(data) /batch_size)
        #将数据按照句子的长度大小从小到大排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        tags = []
        true_lengths=[]  #用于保存该句话的实际长度
        #取该batch中句子的最大长度
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, tag = line
            padding = [0] * (max_length - len(string))  #将少于句子最大长度的部分用0填充
            #分别将小于最大长度的句子的四部分用0填充
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            tags.append(tag + padding)
            true_lengths.append(len(string))
        return [strings, chars, segs, tags,true_lengths]

    def iter_batch(self, shuffle=False):
        if shuffle:
            #将self.batch_data随机混排
            random.shuffle(self.batch_data)
        for idx in range(self.num_batch):
            yield self.batch_data[idx]  #选择其中的一个batch进行训练
