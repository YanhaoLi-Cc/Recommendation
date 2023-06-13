from base.graphRecommender import GraphRecommender
import tensorflow as tf
from util import config
from util.loss import bpr_loss
import numpy as np
import scipy.sparse as sp
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HeavyGCL(GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(HeavyGCL, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(HeavyGCL, self).readConfiguration()
        args = config.OptionConf(self.config['HeavyGCL'])
        self.ssl_reg1 = float(args['-lambda1'])
        self.ssl_reg2 = float(args['-lambda2'])
        self.ssl_reg3 = float(args['-lambda3'])
        self.drop_rate = float(args['-droprate'])
        # self.aug_type = int(args['-augtype'])
        self.ssl_temp = float(args['-temp'])
        self.n_layers = int(args['-n_layer'])
        self.eps = float(args['-eps'])

    def perturbed_LightGCN_encoder(self, emb, adj, n_layers):
        hd_all_embs = []
        for k in range(n_layers):
            emb = tf.sparse_tensor_dense_matmul(adj, emb)
            random_noise = tf.random.uniform(emb.shape)
            emb += tf.multiply(tf.sign(emb), tf.nn.l2_normalize(random_noise, 1)) * self.eps
            hd_all_embs.append(emb)
        all_embs = tf.reduce_mean(hd_all_embs, axis=0)
        return tf.split(all_embs, [self.num_users, self.num_items], 0)

    def initModel(self):
        super(HeavyGCL, self).initModel()
        self.norm_adj = self.create_joint_sparse_adj_tensor()
        norm_adj = self._create_adj_mat(is_subgraph=False)
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        # nd
        nd_s1_embeddings = ego_embeddings
        nd_s2_embeddings = ego_embeddings
        nd_all_s1_embeddings = [nd_s1_embeddings]
        nd_all_s2_embeddings = [nd_s2_embeddings]
        # ed
        ed_s1_embeddings = ego_embeddings
        ed_s2_embeddings = ego_embeddings
        ed_all_s1_embeddings = [ed_s1_embeddings]
        ed_all_s2_embeddings = [ed_s2_embeddings]
        self.perturbed_user_embeddings1, self.perturbed_item_embeddings1 = self.perturbed_LightGCN_encoder(
            ego_embeddings, self.norm_adj, self.n_layers)
        self.perturbed_user_embeddings2, self.perturbed_item_embeddings2 = self.perturbed_LightGCN_encoder(
            ego_embeddings, self.norm_adj, self.n_layers)

        all_embeddings = [ego_embeddings]

        # variable initialization
        self._create_variable()
        for k in range(0, self.n_layers):
            # Node Drop
            self.sub_mat['nd_sub_mat_1%d' % k] = tf.SparseTensor(
                self.sub_mat['adj_indices_nd_sub1'],
                self.sub_mat['adj_values_nd_sub1'],
                self.sub_mat['adj_shape_nd_sub1'])
            self.sub_mat['nd_sub_mat_2%d' % k] = tf.SparseTensor(
                self.sub_mat['adj_indices_nd_sub2'],
                self.sub_mat['adj_values_nd_sub2'],
                self.sub_mat['adj_shape_nd_sub2'])
            # Edge Drop
            self.sub_mat['ed_sub_mat_1%d' % k] = tf.SparseTensor(
                self.sub_mat['adj_indices_ed_sub1'],
                self.sub_mat['adj_values_ed_sub1'],
                self.sub_mat['adj_shape_ed_sub1'])
            self.sub_mat['ed_sub_mat_2%d' % k] = tf.SparseTensor(
                self.sub_mat['adj_indices_ed_sub2'],
                self.sub_mat['adj_values_ed_sub2'],
                self.sub_mat['adj_shape_ed_sub2'])

        # nd - s1 - view
        for k in range(self.n_layers):
            nd_s1_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['nd_sub_mat_1%d' % k], nd_s1_embeddings)
            nd_all_s1_embeddings += [nd_s1_embeddings]
        nd_all_s1_embeddings = tf.stack(nd_all_s1_embeddings, 1)
        nd_all_s1_embeddings = tf.reduce_mean(nd_all_s1_embeddings, axis=1, keepdims=False)
        self.nd_s1_user_embeddings, self.nd_s1_item_embeddings = tf.split(nd_all_s1_embeddings,
                                                                          [self.num_users, self.num_items],
                                                                          0)
        # nd - s2 - view
        for k in range(self.n_layers):
            nd_s2_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['nd_sub_mat_2%d' % k], nd_s2_embeddings)
            nd_all_s2_embeddings += [nd_s2_embeddings]
        nd_all_s2_embeddings = tf.stack(nd_all_s2_embeddings, 1)
        nd_all_s2_embeddings = tf.reduce_mean(nd_all_s2_embeddings, axis=1, keepdims=False)
        self.nd_s2_user_embeddings, self.nd_s2_item_embeddings = tf.split(nd_all_s2_embeddings,
                                                                          [self.num_users, self.num_items],
                                                                          0)

        # ed - s1 - view
        for k in range(self.n_layers):
            ed_s1_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['ed_sub_mat_1%d' % k], ed_s1_embeddings)
            ed_all_s1_embeddings += [ed_s1_embeddings]
        ed_all_s1_embeddings = tf.stack(ed_all_s1_embeddings, 1)
        ed_all_s1_embeddings = tf.reduce_mean(ed_all_s1_embeddings, axis=1, keepdims=False)
        self.ed_s1_user_embeddings, self.ed_s1_item_embeddings = tf.split(ed_all_s1_embeddings,
                                                                          [self.num_users, self.num_items],
                                                                          0)
        # ed - s2 - view
        for k in range(self.n_layers):
            ed_s2_embeddings = tf.sparse_tensor_dense_matmul(self.sub_mat['ed_sub_mat_2%d' % k], ed_s2_embeddings)
            ed_all_s2_embeddings += [ed_s2_embeddings]
        ed_all_s2_embeddings = tf.stack(ed_all_s2_embeddings, 1)
        ed_all_s2_embeddings = tf.reduce_mean(ed_all_s2_embeddings, axis=1, keepdims=False)
        self.ed_s2_user_embeddings, self.ed_s2_item_embeddings = tf.split(ed_all_s2_embeddings,
                                                                          [self.num_users, self.num_items],
                                                                          0)

        # recommendation view
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        self.main_user_embeddings, self.main_item_embeddings = tf.split(all_embeddings,
                                                                        [self.num_users, self.num_items], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.neg_idx)
        self.batch_user_emb = tf.nn.embedding_lookup(self.main_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.main_item_embeddings, self.v_idx)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def _create_variable(self):
        self.sub_mat = {}
        # nd
        self.sub_mat['adj_values_nd_sub1'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_nd_sub1'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_nd_sub1'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_values_nd_sub2'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_nd_sub2'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_nd_sub2'] = tf.placeholder(tf.int64)
        # ed
        self.sub_mat['adj_values_ed_sub1'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_ed_sub1'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_ed_sub1'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_values_ed_sub2'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices_ed_sub2'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape_ed_sub2'] = tf.placeholder(tf.int64)

    def _create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        if is_subgraph and aug_type in [0, 1] and self.drop_rate > 0:
            # nd
            if aug_type == 0:
                drop_user_idx = random.sample(list(range(self.num_users)), int(self.num_users * self.drop_rate))
                drop_item_idx = random.sample(list(range(self.num_items)), int(self.num_items * self.drop_rate))
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            # ed
            if aug_type == 1:
                keep_idx = random.sample(list(range(self.data.elemCount())),
                                         int(self.data.elemCount() * (1 - self.drop_rate)))
                user_np = np.array(row_idx)[keep_idx]
                item_np = np.array(col_idx)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        # hd
        else:
            user_np = np.array(row_idx)
            item_np = np.array(col_idx)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def calc_ssl_loss_v3_nd(self):
        nd_user_emb1 = tf.nn.embedding_lookup(self.nd_s1_user_embeddings, tf.unique(self.u_idx)[0])
        nd_user_emb2 = tf.nn.embedding_lookup(self.nd_s2_user_embeddings, tf.unique(self.u_idx)[0])
        nd_item_emb1 = tf.nn.embedding_lookup(self.nd_s1_item_embeddings, tf.unique(self.v_idx)[0])
        nd_item_emb2 = tf.nn.embedding_lookup(self.nd_s2_item_embeddings, tf.unique(self.v_idx)[0])
        nd_emb_merge1 = tf.concat([nd_user_emb1, nd_item_emb1], axis=0)
        nd_emb_merge2 = tf.concat([nd_user_emb2, nd_item_emb2], axis=0)
        # cosine similarity
        nd_normalize_emb_merge1 = tf.nn.l2_normalize(nd_emb_merge1, 1)
        nd_normalize_emb_merge2 = tf.nn.l2_normalize(nd_emb_merge2, 1)
        nd_pos_score = tf.reduce_sum(tf.multiply(nd_normalize_emb_merge1, nd_normalize_emb_merge2), axis=1)
        nd_ttl_score = tf.matmul(nd_normalize_emb_merge1, nd_normalize_emb_merge2, transpose_a=False, transpose_b=True)
        nd_pos_score = tf.exp(nd_pos_score / self.ssl_temp)
        nd_ttl_score = tf.reduce_sum(tf.exp(nd_ttl_score / self.ssl_temp), axis=1)
        nd_ssl_loss = -tf.reduce_sum(tf.log(nd_pos_score / nd_ttl_score))
        nd_ssl_loss = self.ssl_reg1 * nd_ssl_loss
        return nd_ssl_loss

    def calc_ssl_loss_v3_ed(self):
        ed_user_emb1 = tf.nn.embedding_lookup(self.ed_s1_user_embeddings, tf.unique(self.u_idx)[0])
        ed_user_emb2 = tf.nn.embedding_lookup(self.ed_s2_user_embeddings, tf.unique(self.u_idx)[0])
        ed_item_emb1 = tf.nn.embedding_lookup(self.ed_s1_item_embeddings, tf.unique(self.v_idx)[0])
        ed_item_emb2 = tf.nn.embedding_lookup(self.ed_s2_item_embeddings, tf.unique(self.v_idx)[0])
        ed_emb_merge1 = tf.concat([ed_user_emb1, ed_item_emb1], axis=0)
        ed_emb_merge2 = tf.concat([ed_user_emb2, ed_item_emb2], axis=0)
        # cosine similarity
        ed_normalize_emb_merge1 = tf.nn.l2_normalize(ed_emb_merge1, 1)
        ed_normalize_emb_merge2 = tf.nn.l2_normalize(ed_emb_merge2, 1)
        ed_pos_score = tf.reduce_sum(tf.multiply(ed_normalize_emb_merge1, ed_normalize_emb_merge2), axis=1)
        ed_ttl_score = tf.matmul(ed_normalize_emb_merge1, ed_normalize_emb_merge2, transpose_a=False, transpose_b=True)
        ed_pos_score = tf.exp(ed_pos_score / self.ssl_temp)
        ed_ttl_score = tf.reduce_sum(tf.exp(ed_ttl_score / self.ssl_temp), axis=1)
        ed_ssl_loss = -tf.reduce_sum(tf.log(ed_pos_score / ed_ttl_score))
        ed_ssl_loss = self.ssl_reg2 * ed_ssl_loss
        return ed_ssl_loss

    def calc_ssl_loss_v3_hd(self):
        p_user_emb1 = tf.nn.embedding_lookup(self.perturbed_user_embeddings1, tf.unique(self.u_idx)[0])
        p_item_emb1 = tf.nn.embedding_lookup(self.perturbed_item_embeddings1, tf.unique(self.v_idx)[0])
        p_user_emb2 = tf.nn.embedding_lookup(self.perturbed_user_embeddings2, tf.unique(self.u_idx)[0])
        p_item_emb2 = tf.nn.embedding_lookup(self.perturbed_item_embeddings2, tf.unique(self.v_idx)[0])
        # group contrast
        normalize_emb_user1 = tf.nn.l2_normalize(p_user_emb1, 1)
        normalize_emb_user2 = tf.nn.l2_normalize(p_user_emb2, 1)
        normalize_emb_item1 = tf.nn.l2_normalize(p_item_emb1, 1)
        normalize_emb_item2 = tf.nn.l2_normalize(p_item_emb2, 1)
        pos_score_u = tf.reduce_sum(tf.multiply(normalize_emb_user1, normalize_emb_user2), axis=1)
        pos_score_i = tf.reduce_sum(tf.multiply(normalize_emb_item1, normalize_emb_item2), axis=1)
        ttl_score_u = tf.matmul(normalize_emb_user1, normalize_emb_user2, transpose_a=False, transpose_b=True)
        ttl_score_i = tf.matmul(normalize_emb_item1, normalize_emb_item2, transpose_a=False, transpose_b=True)

        pos_score_u = tf.exp(pos_score_u / 0.2)
        ttl_score_u = tf.reduce_sum(tf.exp(ttl_score_u / 0.2), axis=1)
        pos_score_i = tf.exp(pos_score_i / 0.2)
        ttl_score_i = tf.reduce_sum(tf.exp(ttl_score_i / 0.2), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score_u / ttl_score_u)) - tf.reduce_sum(tf.log(pos_score_i / ttl_score_i))

        return self.ssl_reg3 * cl_loss


    def trainModel(self):
        # main task: recommendation
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (
                tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_pos_item_emb) + tf.nn.l2_loss(
            self.batch_neg_item_emb))
        # SSL task: contrastive learning
        ssl_loss_nd = self.calc_ssl_loss_v3_nd()
        ssl_loss_ed = self.calc_ssl_loss_v3_ed()
        ssl_loss_hd = self.calc_ssl_loss_v3_hd()
        total_loss = rec_loss + ssl_loss_nd + ssl_loss_ed + ssl_loss_hd

        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(total_loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # import time
        for epoch in range(self.maxEpoch):
            sub_mat = {}
            sub_mat['adj_indices_nd_sub1'], sub_mat['adj_values_nd_sub1'], sub_mat[
                'adj_shape_nd_sub1'] = self._convert_csr_to_sparse_tensor_inputs(
                self._create_adj_mat(is_subgraph=True, aug_type=0))
            sub_mat['adj_indices_nd_sub2'], sub_mat['adj_values_nd_sub2'], sub_mat[
                'adj_shape_nd_sub2'] = self._convert_csr_to_sparse_tensor_inputs(
                self._create_adj_mat(is_subgraph=True, aug_type=0))

            sub_mat['adj_indices_ed_sub1'], sub_mat['adj_values_ed_sub1'], sub_mat[
                'adj_shape_ed_sub1'] = self._convert_csr_to_sparse_tensor_inputs(
                self._create_adj_mat(is_subgraph=True, aug_type=1))
            sub_mat['adj_indices_ed_sub2'], sub_mat['adj_values_ed_sub2'], sub_mat[
                'adj_shape_ed_sub2'] = self._convert_csr_to_sparse_tensor_inputs(
                self._create_adj_mat(is_subgraph=True, aug_type=1))

            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                feed_dict = {self.u_idx: user_idx,
                             self.v_idx: i_idx,
                             self.neg_idx: j_idx, }
                feed_dict.update({
                    self.sub_mat['adj_values_nd_sub1']: sub_mat['adj_values_nd_sub1'],
                    self.sub_mat['adj_indices_nd_sub1']: sub_mat['adj_indices_nd_sub1'],
                    self.sub_mat['adj_shape_nd_sub1']: sub_mat['adj_shape_nd_sub1'],
                    self.sub_mat['adj_values_nd_sub2']: sub_mat['adj_values_nd_sub2'],
                    self.sub_mat['adj_indices_nd_sub2']: sub_mat['adj_indices_nd_sub2'],
                    self.sub_mat['adj_shape_nd_sub2']: sub_mat['adj_shape_nd_sub2'],

                    self.sub_mat['adj_values_ed_sub1']: sub_mat['adj_values_ed_sub1'],
                    self.sub_mat['adj_indices_ed_sub1']: sub_mat['adj_indices_ed_sub1'],
                    self.sub_mat['adj_shape_ed_sub1']: sub_mat['adj_shape_ed_sub1'],
                    self.sub_mat['adj_values_ed_sub2']: sub_mat['adj_values_ed_sub2'],
                    self.sub_mat['adj_indices_ed_sub2']: sub_mat['adj_indices_ed_sub2'],
                    self.sub_mat['adj_shape_ed_sub2']: sub_mat['adj_shape_ed_sub2'],
                })
                _, l, rec_l, ssl_l_nd, ssl_l_ed, ssl_l_hd = self.sess.run([train, total_loss, rec_loss, ssl_loss_ed, ssl_loss_hd, ssl_loss_nd], feed_dict=feed_dict)
                print('training:', epoch + 1, 'batch', n, 'loss:', l, 'rec_loss:', rec_l, 'ssl_loss_nd', ssl_l_nd, 'ssl_loss_ed', ssl_l_ed, 'ssl_loss_hd', ssl_l_hd)
            self.U, self.V = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])
            self.ranking_performance(epoch)
        self.U, self.V = self.bestU, self.bestV

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.main_user_embeddings, self.main_item_embeddings])

    def predictForRanking(self, u):
        'rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
