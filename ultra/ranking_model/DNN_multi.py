from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
from ultra.ranking_model import ActivationFunctions
import ultra.utils


class DNNMulti(BaseRankingModel):
    """The deep neural network model for learning to rank.

    This class implements a deep neural network (DNN) based ranking model. It's essientially a multi-layer perceptron network.

    """

    def __init__(self, hparams_str):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Number of neurons in each layer of a ranking_model.
            hidden_layer_sizes=[512, 256, 128, 22],
            # Type for activation function, which could be elu, relu, sigmoid,
            # or tanh
            activation_func='elu',
            initializer='None',
            dropout=0.0,
            keep_dropout_in_inference=False,
            norm="layer"
        )
        self.hparams.parse(hparams_str)
        print("DNN_multi hparam", self.hparams.to_json())
        self.initializer = None
        self.act_func = None
        self.layer_norm = None

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

        self.model_parameters = {}
    
    def build_one_model(self, input_list, model_index, is_training=False, **kwargs):
        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            if "input_tensor" in kwargs:
                print("Use input_tensor!!")
                input_data = kwargs['input_tensor']
            else:
                print("Not Use input_tensor!!")
                input_data = tf.concat(input_list, axis=0) # (T*B, F)
            output_data = input_data
            output_sizes = self.hparams.hidden_layer_sizes

            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                self.layer_norm = []
                for j in range(len(output_sizes)):
                    self.layer_norm.append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                        name="layer_norm_%d" % j))

            current_size = output_data.get_shape()[-1].value
            for j in range(len(output_sizes)):
                if self.layer_norm is not None:
                    if self.hparams.norm == "layer":
                        output_data = self.layer_norm[j](
                            output_data)
                    else:
                        output_data = self.layer_norm[j](
                            output_data, training=is_training)
                expand_W = tf.get_variable("dnn_W_%d_%d" % (model_index, j), [current_size, output_sizes[j]])
                expand_b = tf.get_variable("dnn_b_%d_%d" % (model_index, j), [output_sizes[j]])
                
                output_data = tf.nn.bias_add(
                    tf.matmul(output_data, expand_W), expand_b)
                # Add activation if it is a hidden layer
                if j != len(output_sizes) - 1:
                    output_data = self.act_func(output_data)

                    # if self.hparams.keep_dropout_in_inference:
                    #     is_training = True

                    # if self.hparams.dropout > 0 and "force_use_dropout" not in kwargs:
                    #     output_data = tf.layers.dropout(output_data, rate=self.hparams.dropout,
                    #                                     training=is_training)
                current_size = output_sizes[j]

            # output_data: # (T*B, 20)
            return tf.split(output_data, len(input_list), axis=0) # (T, B, 20)
    
    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the DNN model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents. （T, B, F)
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """

        return self.build_one_model(input_list, 0, is_training, **kwargs)

