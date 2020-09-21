from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class DySAT(Model):
    def _accuracy(self):
        pass

    def __init__(self, placeholders, SGC_weight, motifs_weight, use_motifs, num_features, num_features_nonzero, degrees, min_t, num_time_steps, **kwargs):
        super(DySAT, self).__init__(**kwargs)
        self.attn_wts_all = []
        self.temporal_attention_layers = []
        self.structural_attention_layers = []
        self.placeholders = placeholders
        if FLAGS.window < 0:
            self.num_time_steps = len(placeholders['features'])
        else:
            self.num_time_steps = min(len(placeholders['features']), FLAGS.window + 1)  # window = 0 => only self.
        self.num_time_steps_train = self.num_time_steps - 1
        self.num_features = num_features
        self.num_features_nonzero = num_features_nonzero
        self.degrees = degrees
        self.num_features = num_features
        self.structural_head_config = map(int, FLAGS.structural_head_config.split(","))
        self.structural_layer_config = map(int, FLAGS.structural_layer_config.split(","))
        self.temporal_head_config = map(int, FLAGS.temporal_head_config.split(","))
        self.temporal_layer_config = map(int, FLAGS.temporal_layer_config.split(","))
        # self.motifs = motifs
        self.motifs = [placeholders['motifs{}'.format(t)] for t in range(min_t, num_time_steps)]
        self.SGC_weight = SGC_weight
        self.motifs_weight = motifs_weight
        self.use_motifs = use_motifs
        self._build()

    def _build(self):
        proximity_labels = [tf.expand_dims(tf.cast(self.placeholders['node_2'][t], tf.int64), 1)
                            for t in range(0, len(self.placeholders['features']))]  # [B, 1]

        self.proximity_neg_samples = []
        for t in range(len(self.placeholders['features']) - 1 - self.num_time_steps_train,
                       len(self.placeholders['features']) - 1):
            self.proximity_neg_samples.append(tf.nn.fixed_unigram_candidate_sampler(
                true_classes=proximity_labels[t],
                num_true=1,
                num_sampled=FLAGS.neg_sample_size,
                unique=False,
                range_max=len(self.degrees[t]),
                distortion=0.75,
                unigrams=self.degrees[t].tolist())[0])

        # Build actual model.
        if self.use_motifs:
            self.final_output_embeddings, self.adj_out, self.motif_out, self.motif_weight_out = self.build_net(self.structural_head_config, self.structural_layer_config,
                                                      self.temporal_head_config,
                                                      self.temporal_layer_config,
                                                      self.placeholders['spatial_drop'],
                                                      self.placeholders['temporal_drop'],
                                                      self.placeholders['adjs'],
                                                      self.motifs,
                                                      self.use_motifs)
        else:
            self.final_output_embeddings = self.build_net(self.structural_head_config, self.structural_layer_config,
                                                      self.temporal_head_config,
                                                      self.temporal_layer_config,
                                                      self.placeholders['spatial_drop'],
                                                      self.placeholders['temporal_drop'],
                                                      self.placeholders['adjs'],
                                                      self.motifs,
                                                      self.use_motifs)
        self._loss()
        self.init_optimizer()

    def build_net(self, attn_head_config, attn_layer_config, temporal_head_config, temporal_layer_config,
                  spatial_drop, temporal_drop, adjs, motifs, use_motifs):
        input_dim = self.num_features
        sparse_inputs = True

        # 1: Structural Attention Layers
        # for i in range(0, len(attn_layer_config)):
        #     if i > 0:
        #         input_dim = attn_layer_config[i - 1]
        #         sparse_inputs = False
        #     self.structural_attention_layers.append(StructuralAttentionLayer(input_dim=input_dim,
        #                                                                      output_dim=attn_layer_config[i],
        #                                                                      n_heads=attn_head_config[i],
        #                                                                      attn_drop=spatial_drop,
        #                                                                      ffd_drop=spatial_drop,
        #                                                                      act=tf.nn.elu,
        #                                                                      sparse_inputs=sparse_inputs,
        #                                                                      residual=False))

        # 1. Bulid SGC Layers
        sgc_degree = 2
        for i in range(0, len(attn_layer_config)):
            self.structural_attention_layers.append(SGCLayer(SGC_weight=self.SGC_weight,
                                                            degree = sgc_degree,
                                                            input_dim=input_dim,
                                                            output_dim = int(FLAGS.structural_layer_config),
                                                            n_heads=attn_head_config[i],
                                                            attn_drop=spatial_drop,
                                                            ffd_drop=spatial_drop,
                                                            act=tf.nn.elu,
                                                            sparse_inputs=sparse_inputs,
                                                            use_motifs=use_motifs,
                                                            residual=False))
        # 2: Temporal Attention Layers
        input_dim = attn_layer_config[-1]
        # for i in range(0, len(temporal_layer_config)):
        #     if i > 0:
        #         input_dim = temporal_layer_config[i - 1]
        #     temporal_layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=temporal_head_config[i],
        #                                             attn_drop=temporal_drop, num_time_steps=self.num_time_steps,
        #                                             residual=False)
        #     self.temporal_attention_layers.append(temporal_layer)

        # 3: Structural Attention forward
        # input_list = self.placeholders['features']  # List of t feature matrices. [N x F]
        # for layer in self.structural_attention_layers:
        #     attn_outputs = []
        #     for t in range(0, self.num_time_steps):
        #         out = layer([input_list[t], adjs[t]])
        #         attn_outputs.append(out)  # A list of [1x Ni x F]
        #     input_list = list(attn_outputs)

        # 3: SGC forward
        weight_fea = tf.get_variable("weight_fea", shape=[self.num_features, int(FLAGS.structural_layer_config)],
                                                dtype=tf.float32,trainable=True)
        weight_fea_final = tf.get_variable("weight_fea_final", shape=[int(FLAGS.structural_layer_config), int(FLAGS.structural_layer_config)],
                                                dtype=tf.float32,trainable=True)

        weight_emb = tf.get_variable("weight_emb", shape=[int(1)],dtype=tf.float32, trainable=True)
        

        input_list = self.placeholders['features']  # List of t feature matrices. [N x F]
        for layer in self.structural_attention_layers:
            attn_outputs = []
            temp_outputs = []
            adj_mat = []
            for t in range(0, self.num_time_steps):
                # motif weighted sum
                if use_motifs:
                    adj_t = adjs[t]
                    motifs_t = motifs[t]
                    node_number = adj_t.dense_shape[0]
                    motif_number = motifs_t[0].dense_shape[0]
                    weighted_sumed_motifs = tf.sparse_concat(0, motifs_t)
                    weighted_sumed_motifs = tf.sparse_reshape(weighted_sumed_motifs, [-1,node_number,node_number]) # [M, N, N]
                    weighted_sumed_motifs = weighted_sumed_motifs.__mul__(tf.transpose(self.motifs_weight)) # [N, N]
                    # weighted_sumed_motifs = weighted_sumed_motifs.__mul__(tf.transpose(self.vars['motif_weight']))
                    weighted_sumed_motifs = tf.sparse_reduce_sum(weighted_sumed_motifs, 0)
                    
                    # no motifs
                    # adj_mat.append(tf.sparse_add(adj_t, weighted_sumed_motifs))

                    # get weighted sumed motifs but DO NOT merger it into adjacency matrix
                    # THEN make adj_mat has T*[2*N*N] element
                    # change the operation of adj * feature (N*N * N*D) TO (2*N*N * 2*N*D)
                    # EACH element is 2*N*N

                    # use concat failed use plain method
                    adj_mat.append([adj_t,weighted_sumed_motifs])
                else:
                    adj_mat = adjs

                ############################ No shift ################################
                ## please change degree = 3 AND activate temporal attention code
                # Input: feature, adj, motifs
                # Output: single layer SGC at one time output

                # out = layer([input_list[t], adjs[t], motifs[t]])
                # attn_outputs.append(out)  # A list of [1x Ni x F]

                ############################ shitf V1 #################################
                ## please change degree = 3 AND deactivate temporal attention code
                # temp_fea = tf.sparse_tensor_dense_matmul(input_list[t], weight_fea)
                # if t > 0:
                #     # input list is feature matrix
                #     # when t > 0 means time is shift to next time
                #     # So swap some last time output feature to current time step to capture the temporal information
                    
                #     fold_div = 0.75

                #     # pad = np.array([['top','down'], ['left','right']])
                #     overall_node = adjs[t].dense_shape[0] # get current node number
                #     new_node = adjs[t].dense_shape[0] - tf.to_int64(tf.shape(out[0])[0])
                #     pad_line = [[0, new_node], [0, 0]]
                #     padded_out = tf.pad(out[0], pad_line) # extend output to new node size
                #     new_dim = tf.to_int64(tf.shape(padded_out)[1]) - tf.to_int64(tf.shape(temp_fea)[1])
                #     # However when node size larger than dim new_dim will be negative
                #     # When new_dim is negative the padded_out should be extend instead of input_list
                #     def true_proc():
                #         pad_line = [[0, 0], [0, new_dim]] # output feature dim is larger than original feature dim
                #         padded_fea = tf.pad(temp_fea, pad_line)
                #         overall_dim = tf.shape(padded_fea)[1]
                #         fold = tf.convert_to_tensor(fold_div, tf.float32) # shift ratio
                #         overall_dim = tf.cast(overall_dim, tf.float32)
                #         shift_dim = tf.cast(overall_dim*fold, tf.int32)
                #         # preserve first fold output feature
                #         mask = tf.pad(tf.ones([overall_node, shift_dim],tf.float32), pad_line)
                #         padded_out_t = tf.multiply(padded_out, tf.cast(mask, padded_out.dtype))
                #         return padded_out_t, padded_fea
                #     def false_proc():
                #         pad_line = [[0, 0], [0, -new_dim]] # output feature dim is larger than original feature dim
                #         padded_fea = temp_fea
                #         padded_out_t = tf.pad(padded_out, pad_line) # padding output to larger dim
                #         overall_dim = tf.shape(padded_fea)[1]
                #         fold = tf.convert_to_tensor(fold_div, tf.float32) # shift ratio
                #         overall_dim = tf.cast(overall_dim, tf.float32)
                #         shift_dim = tf.cast(overall_dim*fold, tf.int32)
                        
                #         temp_dim = tf.to_int32(tf.shape(temp_fea)[1])-shift_dim
                #         pad_line = [[0, 0], [0, temp_dim]]
                #         mask = tf.pad(tf.ones([overall_node, shift_dim],tf.float32), pad_line)
                #         padded_out_t = tf.multiply(padded_out_t, tf.cast(mask, padded_out.dtype))
                #         return padded_out_t, padded_fea
                #     standard = tf.constant(value = 0, dtype = tf.int64)
                #     padded_out, padded_fea = tf.cond(pred = tf.greater(new_dim, standard), true_fn = true_proc, false_fn = false_proc)
                #     out = layer([0.1*padded_out+padded_fea, adjs[t], motifs[t]])
                # else:
                #     # Input: feature, adj, motifs
                #     # Output: single layer SGC at one time output
                #     # Then shift 1/fold features to next time steps
                #     out = layer([temp_fea, adjs[t], motifs[t]])
                # attn_outputs.append(out)  # A list of [1x Ni x F]

                ############################ shitf V2 #################################
                ## please change degree = 1 AND deactivate temporal attention code
                # The first layer of SGC do not need anything to change
                # transform feature to hidden dim
                degree = 10 # means feature through how many times of TIME module

                # use the temp_fea_transform into last layer
                feature_transform_time = time.time()
                temp_fea = tf.sparse_tensor_dense_matmul(input_list[t], weight_fea)
                feature_transform_time = time.time() - feature_transform_time
                print("feature transform time: {:.4f}s".format(feature_transform_time))

                # N*hidden_dim N*128
                # out = layer([input_list[t], adjs[t], motifs[t]])
                # out = layer([temp_fea, adjs[t], motifs[t]])

                # in order use motif and adj matrix, MUST change the input of layer as C*N*N (adj) and C*N*D
                if use_motifs:
                    adj_out = layer([temp_fea, adj_mat[t][0]])
                    motif_out = layer([temp_fea, adj_mat[t][1]])

                    out = tf.add(adj_out, weight_emb*motif_out)
                else:
                    out = layer([temp_fea, adj_mat[t]])

                temp_outputs.append(out)
                if degree == 1:
                    # when in the last layer
                    # out = tf.matmul(out, weight_fea_final)
                    out = tf.expand_dims(out, axis=0) # [1, N, F]
                    attn_outputs.append(out)  # A list of [1x Ni x F]
            for degree_ind in range(degree):
                last_time_outputs = temp_outputs
                temp_outputs = []
                for t in range(0, self.num_time_steps):
                    if t > 0:
                        # temp_outputs is feature matrix
                        # when t > 0 means need shift to next time
                        # So shift part of last time hidden feature to current time step hidden feature to capture the temporal information
                        
                        shift_time = time.time()

                        fold_div = 1 # shift division
                        
                        fold = tf.convert_to_tensor(fold_div, tf.float32) # shift ratio
                        # the hidden dimension is same
                        # the node number is NOT same
                        # just assign last time hidden features value to current time
                        last_node_num = tf.to_int32(tf.shape(temp_outputs[t-1])[0])
                        hidden_dim = tf.cast(tf.to_int32(tf.shape(temp_outputs[t-1])[1]), tf.float32)
                        shift_dim = tf.cast(hidden_dim*fold, tf.int32)

                        # # concat method V1
                        # # shift feature
                        # last_fea = last_time_outputs[t-1]
                        # # last_time_outputs[t-1] is N[t-1]*hidden_dim e.g. 24*128
                        old_fea = last_time_outputs[t-1][:,:shift_dim]
                        # old_fea N[t-1]*shift_dim e.g. 24*12
                        new_fea = last_time_outputs[t][:last_node_num,shift_dim:]
                        # new_fea N[t-1]*(hidden_dim-shift_dim) e.g. 24*(128-12)
                        concated_fea = tf.concat([old_fea, new_fea], 1)
                        # concated_fea N[t-1]*hidden_dim e.g. 24*128
                        rest_new_fea = last_time_outputs[t][last_node_num:,:]
                        # rest_new_fea (N[t]-N[t-1])*hidden_dim e.g. (36-24)*128
                        shifted_fea = tf.concat([concated_fea,rest_new_fea], 0)

                        shift_time = time.time() - shift_time
                        print("each shift time:{:.4f}s".format(shift_time))

                        # output
                        SGC_time = time.time()
                        origin_fea = last_time_outputs[t]
                        
                        if use_motifs:
                            # out = layer([shifted_fea, adj_mat[t]])
                            adj_out = layer([shifted_fea, adj_mat[t][0]])
                            motif_out = layer([origin_fea, adj_mat[t][1]])
                            # motif_out = layer([shifted_fea, adj_mat[t][1]])

                            out = tf.add(adj_out, weight_emb*motif_out)
                        else:
                            out = layer([shifted_fea, adj_mat[t]])

                        SGC_time = time.time() - SGC_time
                        print("each SGC time:{:.4f}s".format(SGC_time))

                        # shift + original into SGC layer
                        # out = layer([shifted_fea+last_time_outputs[t], adjs[t], motifs[t]])

                        # after shifted SGC layer + original
                        # out = out + last_time_outputs[t]
                        
                        # # pad method V2
                        # old_fea = last_time_outputs[t-1][:,:shift_dim]
                        # new_node = adjs[t].dense_shape[0] - adjs[t-1].dense_shape[0]
                        # rest_dim = tf.cast(hidden_dim, tf.int32) - shift_dim
                        # pad_line = [[0, new_node], [0, rest_dim]]
                        # padded_fea = tf.pad(old_fea, pad_line)
                        # shifted_fea = last_time_outputs[t] + padded_fea
                        # out = layer([shifted_fea, adjs[t], motifs[t]])
                    
                    else:
                        # Input: feature, adj, motifs
                        # Output: single layer SGC at one time output
                        # Then shift 1/fold features to next time steps
                        # out = layer([last_time_outputs[t], adjs[t], motifs[t]])
                        temp_fea = last_time_outputs[t]
                        if use_motifs:
                            adj_out = layer([last_time_outputs[t], adj_mat[t][0]])
                            motif_out = layer([last_time_outputs[t], adj_mat[t][1]])
                            out = tf.add(adj_out, weight_emb*motif_out)
                        else:
                            out = layer([last_time_outputs[t], adj_mat[t]])
                    temp_outputs.append(out)
                    if degree_ind == degree-1:
                        # when in the last layer
                        # out = tf.matmul(out, weight_fea_final)
                        out = tf.expand_dims(out, axis=0) # [1, N, F]
                        attn_outputs.append(out)  # A list of [1x Ni x F]
            
            input_list = list(attn_outputs)

        # 4: Pack embeddings across snapshots.
        for t in range(0, self.num_time_steps):
            zero_padding = tf.zeros(
                [1, tf.shape(attn_outputs[-1])[1] - tf.shape(attn_outputs[t])[1], attn_layer_config[-1]])
            attn_outputs[t] = tf.concat([attn_outputs[t], zero_padding], axis=1)
            # match node number to last time step node number

        structural_outputs = tf.transpose(tf.concat(attn_outputs, axis=0), [1, 0, 2])  # [N, T, F]
        structural_outputs = tf.reshape(structural_outputs,
                                        [-1, self.num_time_steps, attn_layer_config[-1]])  # [N, T, F]

        # 5: Temporal Attention forward
        # temporal_inputs = structural_outputs
        # for temporal_layer in self.temporal_attention_layers:
        #     outputs = temporal_layer(temporal_inputs)  # [N, T, F]
        #     temporal_inputs = outputs
        #     self.attn_wts_all.append(temporal_layer.attn_wts_all)
        # return outputs
        if use_motifs:
            return structural_outputs, adj_out, motif_out, self.motifs_weight
        else:
            return structural_outputs

    def _loss(self):
        # loss time
        loss_time = time.time()

        self.graph_loss = tf.constant(0.0)
        num_time_steps_train = self.num_time_steps_train
        for t in range(self.num_time_steps_train - num_time_steps_train, self.num_time_steps_train):
            # fix Converting sparse IndexedSlices to a dense Tensor fo unknown shape
            # temp_emb = tf.transpose(self.final_output_embeddings, [1, 0, 2])
            # self.inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.num_time_steps_train-(self.num_time_steps_train - num_time_steps_train),
            #                                      dynamic_size=False, infer_shape=True)
            # self.inputs_ta = self.inputs_ta.unstack(temp_emb)
            # output_embeds_t = self.inputs_ta.read(t)

            # output_emb = tf.TensorArray(dtype=tf.float32, size=0,
            #                                      dynamic_size=True, infer_shape=True)
            # output_emb = output_emb.unstack(output_embeds_t)
            # # output_emb = output_emb.write(output_embeds_t)

            # inputs1 = output_emb.read(self.placeholders['node_1'][t])
            # inputs2 = output_emb.read(self.placeholders['node_2'][t])
            # pos_score = tf.reduce_sum(inputs1 * inputs2, axis=1)
            # neg_samples = output_emb.read(self.proximity_neg_samples[t])

            # origin
            output_embeds_t = tf.nn.embedding_lookup(tf.transpose(self.final_output_embeddings, [1, 0, 2]), t)
            inputs1 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_1'][t])
            inputs2 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_2'][t])
            pos_score = tf.reduce_sum(inputs1 * inputs2, axis=1)
            neg_samples = tf.nn.embedding_lookup(output_embeds_t, self.proximity_neg_samples[t])
            neg_score = (-1.0) * tf.matmul(inputs1, tf.transpose(neg_samples))
            pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score)
            neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(neg_score), logits=neg_score)
            self.graph_loss += tf.reduce_mean(pos_ent) + FLAGS.neg_weight * tf.reduce_mean(neg_ent)

        self.reg_loss = tf.constant(0.0)
        if len([v for v in tf.trainable_variables() if "struct_attn" in v.name and "bias" not in v.name]) > 0:
            self.reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                       if "struct_attn" in v.name and "bias" not in v.name]) * FLAGS.weight_decay
        self.loss = self.graph_loss + self.reg_loss
        loss_time = time.time() - loss_time
        print('loss time:{:.4f}s'.format(loss_time))

    def init_optimizer(self):
        trainable_params = tf.trainable_variables()
        actual_loss = self.loss
        gradients = tf.gradients(actual_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))
