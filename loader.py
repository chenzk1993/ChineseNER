import re,os
from data_utils import iob2, iob2iobes

def create_dict(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    Dict = {}
    for items in item_list:
        for item in items:
            if item not in Dict:
                Dict[item] = 1
            else:
                Dict[item] += 1
    return Dict

def create_mapping(Dict):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    #对每个汉字(or NEG)出现的次数按从大到小排序
    sorted_items = sorted(Dict.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    #id_to_item为字典，键为序号，值为出现次数从大到小的汉字(or NEG)
    item_to_id = {v: k for k, v in id_to_item.items()}
    #item_to_id为字典，键为出现次数从大到小的汉字(or NEG)，值为序号
    return item_to_id, id_to_item

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in open(path, 'r', encoding='utf8'):
        num+=1
        #当zeros为True时，表示用0代替字符串中的数字
        #.rstrip()   移除字符串右侧的空格
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()  #将字符串从空格处分开
            else:
                word= line.split()
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            #此时命名实体类型只有BOI三种
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            #此时命名实体类型有B-LOC(实体开始),I_LOC(实体内部),E_LOC(实体结束),S_LOC(一个字构成的实体),O等
            new_tags = iob2iobes(tags)  #将标签IOB转化为IOBES
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    #chars为每句话字构成的列表
    Dict_char = create_dict(chars)
    #Dict为一个字典，键为每个汉字，值为该汉字在这段语料库中出现的次数
    Dict_char["<PAD>"] = 10000001
    Dict_char['<UNK>'] = 10000000
    
    #char_to_id为字典，键为出现次数从大到小的汉字，值为序号
    #id_to_char为字典，键为序号，值为出现次数从大到小的汉字
    char_to_id, id_to_char = create_mapping(Dict_char)
    
    print("Found %i unique words (%i in total)" % (len(Dict_char), sum(len(x) for x in chars)))
    return Dict_char, char_to_id, id_to_char

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    #chars为每句话中每个字的NER状态构成的列表
    Dict_tag = create_dict(tags)
    #Dict_tag 为一个字典，键为每个字的NER状态(O,B-LOC,I-LOC,B-PER,I-LOC等)，值为该状态在这段语料库中出现的次数
    tag_to_id, id_to_tag = create_mapping(Dict_tag) 
    #tag_to_id为字典，键为NER状态，值为序号
    #id_to_tag为字典，键为序号，值为NER状态
    print("Found %i unique named entity tags" % len(Dict_tag ))
    return Dict_tag , tag_to_id, id_to_tag

#此函数的作用是在训练集汉字的基础上，添加一些ext_emb_path文件里的汉字
def augment_with_pretrained(dictionary, ext_emb_path, chars):
    #chars为一维列表，每个元素为一个汉字
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding='utf-8')
        if len(ext_emb_path) > 0])
    #pretrained为词向量文件里的汉字构成的set集合
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            #当x在pretrained中，且x也在chars中但x不在dictionary中，则将x作为键添加到dictionary中，
            #其值对应为0
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0
    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word




