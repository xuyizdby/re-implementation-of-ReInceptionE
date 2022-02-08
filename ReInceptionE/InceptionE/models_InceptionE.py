#coding:utf-8
import tensorflow as tf
from collections import defaultdict


def conv2d(inputs,kernel_size,padding="SAME",activation_fn=tf.nn.relu,name="conv",use_bias=True):
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable(name="weights", shape=kernel_size,initializer=tf.initializers.he_uniform())
        outputs = tf.compat.v1.nn.conv2d(inputs, filter=W, strides=[1, 1, 1, 1], padding=padding)
        if use_bias:
            b = tf.Variable(tf.constant(0.0, shape=[kernel_size[-1]]), name="b")
            outputs+=b
        outputs=activation_fn(outputs)
    return outputs

class InceptionE(object):
    def __init__(self,params,entity_embedding,relation_embedding):
        self.init_params(params,entity_embedding,relation_embedding)
        self.name=self.__class__.__name__
        self.features=self.build_placeholder()
        self.build_model(self.features)


    def build_placeholder(self):
        features=defaultdict(lambda :defaultdict(dict))
        #triples
        features["inputs"]["head"]=tf.compat.v1.placeholder(dtype=tf.int32,shape=[None],name="head")
        features["inputs"]["tail"]=tf.compat.v1.placeholder(dtype=tf.int32,shape=[None],name="tail")
        features["inputs"]["relation"]=tf.compat.v1.placeholder(dtype=tf.int32,shape=[None],name="relation")
        self.keep_prob=features["keep_prob"]=tf.compat.v1.placeholder(dtype=tf.float32,name="keep_prob")
        self.lr=tf.compat.v1.placeholder(dtype=tf.float32,name="keep_prob")
        return features


    def build_model(self,features):
        #predict scores
        scores_predict=self.forward(features)
        self.scores = scores_predict
        self.scores_predict=scores_predict
        targets=features["inputs"]["tail"]

        self.entity_embeddings=self.entity_embedding_weights
        self.relation_embeddings=self.relation_embedding_weights

        #compute loss
        self.loss=self.compute_loss_softmax(scores_predict,targets)
        if self.l2_reg_lambda:
            #l2 loss
            l2_loss=tf.add_n(tf.compat.v1.get_collection("l2"))
            self.loss=self.loss+l2_loss*self.l2_reg_lambda

        self.train_op=self.get_train_op(self.loss,learning_rate=self.lr)
        self.predict_outputs={"output":self.scores,"loss":self.loss}
        self.train_outputs={"train_op":self.train_op,"loss":self.loss,"output":self.scores}

    def init_params(self,params,entity_embedding,relation_embedding):
        self.init_entity_embedding=entity_embedding
        self.init_relation_embedding=relation_embedding
        self.entity_embedding_weights=None
        self.relation_embedding_weights = None
        self.share_emb = params.share_emb

        #model size
        self.entity_vocab_size=params.entity_vocab_size
        self.relation_vocab_size=params.relation_vocab_size
        self.emb_dim=params.emb_dim
        self.hidden_dim=params.hidden_dim
        self.num_filter=32
        if self.init_entity_embedding is not None:
            assert self.init_entity_embedding.shape[1]==self.emb_dim,(self.init_entity_embedding.shape,self.emb_dim)
        #train
        self.l2_reg_lambda=params.l2_reg_lambda or 0
        self.optimizer=params.optimizer
        self.gamma = params.gamma or 1

        with tf.compat.v1.variable_scope("embdding"):
            if self.init_entity_embedding is not None:
                print("init embedding from pretrained...")
                entity_embeddings=tf.compat.v1.get_variable(name="entity_embeddings",
                                                  initializer=self.init_entity_embedding,
                                                  dtype=tf.float32)
                target_entity_embeddings=tf.compat.v1.get_variable(name="target_entity_embeddings",
                                                         initializer=self.init_entity_embedding,
                                                          dtype=tf.float32)
                relation_embeddings= tf.compat.v1.get_variable(name="relation_embeddings",
                                                     initializer=self.init_relation_embedding,
                                                      dtype=tf.float32)

            else:
                entity_embeddings=tf.compat.v1.get_variable(name="entity_embeddings",
                                                  shape=[self.entity_vocab_size,self.emb_dim],
                                                  dtype=tf.float32)
                relation_embeddings= tf.compat.v1.get_variable(name="relation_embeddings",
                                                     shape=[self.relation_vocab_size*2,self.emb_dim],
                                                     dtype=tf.float32)
            self.entity_embedding_weights=entity_embeddings
            if self.share_emb:
                self.target_entity_embedding_weights=entity_embeddings
            else:
                self.target_entity_embedding_weights=target_entity_embeddings

            self.relation_embedding_weights=relation_embeddings


    def inceptionE(self,head_emb,relations,num_filter=32):
        head_emb=tf.reshape(head_emb,shape=[-1,10,self.emb_dim//10,1])
        relations=tf.reshape(relations,shape=[-1,10,self.emb_dim//10,1])
        triples=tf.concat([head_emb,relations],axis=-1) #[batch_size,10,10,2]

        with tf.compat.v1.variable_scope("branch_0"):
            branch1x1=conv2d(triples,kernel_size=[1,1,2,num_filter],padding="SAME",activation_fn=tf.nn.relu)
        with tf.compat.v1.variable_scope("branch_1"):
            branch3x3 = conv2d(triples, kernel_size=[1, 1, 2, num_filter], padding="SAME", activation_fn=tf.nn.relu,name="conv0")
            branch3x3 = conv2d(branch3x3, kernel_size=[3, 3, 32, num_filter], padding="SAME", activation_fn=tf.nn.relu,name="conv1_1")
        with tf.compat.v1.variable_scope("branch_2"):
            branch5x5 = conv2d(triples, kernel_size=[1, 1, 2, num_filter], padding="SAME", activation_fn=tf.nn.relu,name="conv0")
            branch5x5 = conv2d(branch5x5, kernel_size=[3, 3, 32, num_filter], padding="SAME", activation_fn=tf.nn.relu,name="conv1_1")
            branch5x5 = conv2d(branch5x5, kernel_size=[3, 3, 32, num_filter], padding="SAME", activation_fn=tf.nn.relu,name="conv1_2")
        with tf.compat.v1.variable_scope("branch_3"):
            branch2x2=conv2d(triples,kernel_size=[1,1,2,num_filter],padding="SAME",activation_fn=tf.nn.relu,name="conv1")
            branch2x2=conv2d(branch2x2,kernel_size=[2,2,32,num_filter],padding="SAME",activation_fn=tf.nn.relu,name="conv2")
        conv_outputs = tf.concat([branch1x1, branch5x5, branch3x3,branch2x2], axis=-1)
        conv_size=num_filter*4

        #vectorization
        conv_outputs=tf.reshape(conv_outputs,shape=[-1,self.emb_dim*conv_size])
        conv_outputs = tf.nn.dropout(conv_outputs, rate=1-self.keep_prob)

        with tf.compat.v1.variable_scope("dense"):
            W=tf.compat.v1.get_variable(name="weights",shape=[self.emb_dim*conv_size,self.emb_dim],initializer=tf.initializers.he_uniform())
            b=tf.compat.v1.get_variable(name="bias",shape=[self.emb_dim],initializer=tf.zeros_initializer)
            outputs=tf.matmul(conv_outputs,W)+b
            tf.compat.v1.add_to_collection(name="l2", value=tf.nn.l2_loss(W))
            tf.compat.v1.add_to_collection(name="l2", value=tf.nn.l2_loss(b))
            outputs=tf.nn.relu(outputs)
            outputs=tf.nn.dropout(outputs,rate=1-self.keep_prob)

        return outputs

    def forward(self,features):
        #query
        heads = features["inputs"]["head"]
        relations = features["inputs"]["relation"]
        #tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
        query_head_emb = tf.nn.embedding_lookup(self.entity_embedding_weights, heads)
        query_relation_emb = tf.nn.embedding_lookup(self.relation_embedding_weights, relations)

        tf.compat.v1.add_to_collection("l2", tf.nn.l2_loss(query_head_emb))
        tf.compat.v1.add_to_collection("l2", tf.nn.l2_loss(query_relation_emb))

        with tf.compat.v1.variable_scope("hr_nodes_emb"):
            query_emb = self.inceptionE(query_head_emb, query_relation_emb,num_filter=self.num_filter)
            query_emb = tf.nn.dropout(query_emb, rate=1-self.keep_prob)
        with tf.compat.v1.variable_scope("scores"):
            scores_predict = tf.matmul(query_emb, self.target_entity_embedding_weights, transpose_b=True)
        return scores_predict

    def prepare_for_GAT(self,features):
        #query
        heads = features["inputs"]["head"]
        relations = features["inputs"]["relation"]
        #tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
        query_head_emb = tf.nn.embedding_lookup(self.entity_embedding_weights, heads)
        query_relation_emb = tf.nn.embedding_lookup(self.relation_embedding_weights, relations)

        tf.compat.v1.add_to_collection("l2", tf.nn.l2_loss(query_head_emb))
        tf.compat.v1.add_to_collection("l2", tf.nn.l2_loss(query_relation_emb))

        with tf.compat.v1.variable_scope("hr_nodes_emb"):
            query_emb = self.inceptionE(query_head_emb, query_relation_emb,num_filter=self.num_filter)
            query_emb = tf.nn.dropout(query_emb, rate=1-self.keep_prob)
        return query_emb

    def GAT(self,features):
        #query v_q:
        query_emb=self.prepare_for_GAT(features)
        #local attention
        neighbor_emb = self.prepare_for_GAT(features)

        #global attention
        #head
        global_head_emb = self.prepare_for_GAT(features)
        #tail
        global_tail_emb = self.prepare_for_GAT(features)

    def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation

    # loss function
    def compute_loss_softmax(self,scores,labels):
        scores=self.gamma*scores
        losses=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels,scores,reduction="none")
        loss=tf.reduce_mean(losses)
        return loss

    def get_train_op(self,loss,learning_rate=0.001):
        optimizer=self.optimizer(learning_rate)
        var_list=tf.compat.v1.trainable_variables()
        grad_vars=optimizer.compute_gradients(loss,var_list=var_list,aggregation_method=2)
        grad_vars=[(tf.clip_by_value(g,clip_value_min=-5,clip_value_max=5),v) for g,v in grad_vars if g is not None]
        train_op=optimizer.apply_gradients(grad_vars)
        return train_op

