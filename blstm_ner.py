# encoding = utf8
import numpy as np
import tensorflow as tf
from data_utils import iobes2iob,load_word2vec
class BLSTM_Model(object):
    def __init__(self,config,logger):
        self.config = config
        self.logger=logger
        self.learning_rate = config["learning_rate"]
        self.char_dim = config["char_dim"]   #100
        self.seg_dim = config["seg_dim"]    #20
        self.lstm_dim = config["lstm_dim"]  #100
        
        self.num_tags = config["num_tags"]    #13
        self.num_chars = config["num_chars"]  #4313 
        self.num_segs = 4  #['0','1,3','1,2,3','1,2,2,3']

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        #对权重进行行“Xavier”初始化，此种方式目的使初始化深度学习网络的时候让权重不大不小。
        #变量范围为`x = sqrt(6. / (in + out)); [-x, x]` 
        self.initializer = tf.contrib.layers.xavier_initializer()

        # add placeholders for the model
        #shape=[batch_size,该批次最长句子的长度 ]
        self.char_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None],name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None],name="SegInputs")
        self.true_tag_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None],name="TagInputs")
#        #保存该batch中每句话的实际长度
#        self.true_lengths = tf.placeholder(dtype=tf.int32,shape=[None],name="true lengths")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,name="Dropout")
        
        used=tf.ones_like(self.char_inputs) 
        #reduction_indices=1意味着将d纵向缩减，即按行求和
        #reduction_indices=0意味着将d横向缩减，即按列求和
        length = tf.reduce_sum(used, reduction_indices=1) #如[[32],[32],[32],[32],[32],[32],[32],[32]]
        #将length类型转化为tf.int32类型
        #sequence_lengths为一个二维张量，如[[32],[32],[32],[32],[32],[32],[32],[32]]
        #其中一维的大小为该batch训练数据的长度，32表示每一行中序列的长度
        self.sequence_lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]  #获取self.char_inputs的行数
        #注意max_seq_len为self.char_inputs的列数即每句话中汉字的个数
        self.max_seq_len = tf.shape(self.char_inputs)[-1]  
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        #embedding包含两部分[char_embedding,seg_embedding]
        #char_embedding形状为batch_size*该批次最长句子的长度*char_dim 
        #seg_embedding形状为batch_size*该批次最长句子的长度*seg_dim
        # apply dropout before feed to lstm layer
        self.lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer()

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)
        # loss of the model
        self.loss = self.loss_layer()

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def build_graph(self,session,id_to_char):
        self.logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if self.config["pre_emb"]:
            #返回变量char_lookup的值，在当前上下文中读取。
            emb_weights = session.run(self.char_lookup.read_value())
            #加载训练好的词向量
            emb_weights = load_word2vec(self.config["emb_file"],id_to_char, 
                                                      self.config["char_dim"], emb_weights)
            #为char_lookup变量分配一个新值
            session.run(self.char_lookup.assign(emb_weights))
            self.logger.info("Load pre-trained embedding.")
    
    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: char feature
        :param seg_inputs: segmentation feature
        :param config: whether use segmentation feature
        :return: [batch_size, max_seq_len, embedding_size], 
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            #当该变量存在时，则获得该变量;当该变量不存在时，则和tf.Variable()一样，创建该变量
            #当trainable=True(默认值)时表示，每个字的词向量会在训练过程中自动更新
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim], trainable=True,
                    initializer=self.initializer)
#            word_embeddings = tf.Variable(self.embeddings,
#                                           dtype=tf.float32,
#                                           trainable=self.update_embedding,
#                                           name="word_embeddings")
            #embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
#tf.concat(concat_dim, values, name='concat')concat_dim是tensor连接的方向（维度），
#values是要连接的tensor链表，name是操作名。两个二维tensor连接：0表示按行连接，增加行数，列数不变;
#1表示按列连接，增加列数。两个三维tensor连接:concat_dim：0表示纵向，1表示行，2表示列
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, name=None):
        """
        :param lstm_inputs: [batch_size, max_seq_len, embedding_size] 
        :return: [batch_size, max_seq_len, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
             cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
             cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_dim)
             (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.lstm_inputs,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
             output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
             return output

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, max_seq_len, 2*lstm_dim] 
        :return: [batch_size, max_seq_len, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                #lstm_outputs变成了[batch_size*max_seq_len, 2*lstm_dim] 
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                #tf.nn.xw_plus_b()计算matmul(x, weights) + biases
                #hidden变成了[batch_size*max_seq_len, lstm_dim] 
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                #pred变成了[batch_size*max_seq_len, num_tags] 
                pred = tf.nn.xw_plus_b(hidden, W, b)
            #用-1防止因为最后剩余的一批数据个数不到batch_size而报错，-1会自动根
            #据输出结果的大小调整第一维的大小
            return tf.reshape(pred, [-1, self.max_seq_len, self.num_tags])

    def loss_layer(self):
        """
        calculate crf loss
        :param self.logits: [batch_size, max_seq_len, num_tags]
        """
        with tf.variable_scope("crf_loss"):
            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,tag_indices=self.true_tag_inputs,
                sequence_lengths=self.sequence_lengths)
            return tf.reduce_mean(-log_likelihood)
        
    def decode(self,logits, lengths,trans):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param trans: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        for logit, length in zip(logits, lengths):
            logit = logit[:length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit, trans)
            #viterbi_seq表示的是一句话(补齐到最大长度后的句子)中每个字的NER状态的序号组成的列表
            paths.append(viterbi_seq)
        return paths
    
    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        #chars为一个batch中所有汉字的序号构成的列表;segs为一个batch所有分词后的汉语词汇的序号构成的列表;
        #tags为一个batch中所有汉字对应的NER状态的序号构成的列表,#seq_lens为一个batch中所有句子的实际
        #长度构成的列表
        _, chars, segs, tags,seq_lens = batch
        #np.asarray将列表转化为数组
        #而在测试时只用到了chars,segs两个特征，没用dropout
        feed_dict = {
                self.char_inputs: np.asarray(chars),
                self.seg_inputs: np.asarray(segs),
                self.true_tag_inputs:np.asarray(tags),
                self.dropout: 1.0
        }
        #训练时用到了chars,segs,tags三个特征，和dropout
        if is_train:
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict
    
    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],feed_dict=feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.sequence_lengths, self.logits], feed_dict=feed_dict)
            return lengths, logits

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans=self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]  #string为一批数据中每一句话中的汉字构成的二维列表
            tags = batch[3]    #tags为一批数据中每一句话中的NER状态的序号构成的二维列表
            lengths, logits = self.run_step(sess, False, batch)  #默认非训练状态
            batch_paths = self.decode(logits, lengths,trans)
            #准确率，求和计算算数平均值
            for i in range(len(strings)):
                result = []
                chars = strings[i][:lengths[i]]
                golds = iobes2iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                preds = iobes2iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(chars, golds, preds):
                    result.append(" ".join([str(char), gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_tag):
        trans = self.trans.eval()
        lengths, logits = self.run_step(sess, False, inputs)
        batch_paths= self.decode(logits, lengths,trans)
        tags = [id_tag[idx] for idx in batch_paths[0]]
        return self.printResult(inputs[0][0], tags)
    
    def printResult(self,string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item
