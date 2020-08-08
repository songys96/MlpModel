# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utils.mathUtils import *

class Dataset(object):
    def __init__(self, name, mode):
        """
        mode값을 보관
        예시 : 'regression', 'binary', 'select'
        """
        self.name = name
        self.mode = self.check_mode(mode)

    def __str__(self):
        return f'{self.name}({self.mode}, {len(self.tr_xs)}+{len(self.te_xs)}+{len(self.va_xs)})'

    def check_mode(self, mode):
        if mode not in ['regression', 'binary', 'select']:
            print("임의의 모드입니다 {} (Dataset.py)".format(mode))
            return mode
        return mode

    @property
    def train_count(self):
        """
        @property
        를 통해 메서드이면서 함수가 아닌 속성으로 취급되어
        obj.train_count() 대신 obj.train_count 으로 접근가능해진다.
        self외의 속성값을 인자로 받으면 안됨!!
        """
        return len(self.tr_xs)

    # -----------------------------------------------
    # 데이터 전처리와 관련된 코드입니다

    def get_train_data(self, batch_size, nth):
        """
        훈련데이터를 셔플한 상태로 가져오기
        """
        from_idx = nth * batch_size
        to_idx = (nth + 1) * batch_size

        tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

        return tr_X, tr_Y

    def shuffle_train_data(self, size):
        """
        훈련데이터 섞기
        """
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)

    def get_test_data(self):
        """
        테스트 데이터 가져오기
        """
        return self.te_xs, self.te_ys

    def get_validate_data(self, count):
        """
        검증데이터 섞어서 가져오기
        """
        self.va_indices = np.arange(len(self.va_xs))
        np.random.shuffle(self.va_indices)

        va_X = self.va_xs[self.va_indices[0:count]]
        va_Y = self.va_ys[self.va_indices[0:count]]

        return va_X, va_Y

    def get_visualize_data(self, count):
        """
        검증데이터 섞어서 가져오기
        """
        self.va_indices = np.arange(len(self.va_xs))
        np.random.shuffle(self.va_indices)

        va_X = self.va_xs[self.va_indices[0:count]]
        va_Y = self.va_ys[self.va_indices[0:count]]

        return va_X, va_Y

    def shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        """
        학습, 테스트, 검정 데이터는 모두 이 메서드에서 만들어지며
        위의 메서드들은 MlpModel에서 직접 사용될 예정이다
        """
        data_count = len(xs)

        tr_cnt = int(data_count * tr_ratio / 10) * 10
        va_cnt = int(data_count * va_ratio)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)
        np.random.shuffle(indices)

        # 랜덤한 인섹스를 인덱싱 한 값
        indices_tr = indices[tr_from:tr_to]
        indices_va = indices[va_from:va_to]
        indices_te = indices[te_from:te_to]

        # 위에서 만든 랜덤인덱스를 train, validate, test 데이터에 넣어서 
        # 랜덤한 array값 갖기
        self.tr_xs = xs[indices_tr]
        self.tr_ys = ys[indices_tr]
        self.va_xs = xs[indices_va]
        self.va_ys = ys[indices_va]
        self.te_xs = xs[indices_te]
        self.te_ys = ys[indices_te]

        self.input_shape = xs[0].shape
        self.output_shape = ys[0].shape

        return indices_tr, indices_va, indices_te

    

    # -----------------------------------------------
    # 학습과 관련된 코드입니다

    def forward_postproc(self, output, y, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            diff = output - y
            square = np.square(diff)
            loss = np.mean(square)
            aux = diff
            
        elif mode == 'binary':
            entropy = sigmoid_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [y, output]
        elif mode == 'select':
            entropy = softmax_cross_entropy_with_logits(y, output)
            loss = np.mean(entropy)
            aux = [output, y, entropy]

        return loss, aux

    def backprop_postproc(self, G_loss, aux, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            diff = aux
            shape = diff.shape

            g_loss_square = np.ones(shape) / np.prod(shape)
            g_square_diff = 2 * diff
            g_diff_output = 1 

            G_square = g_loss_square * G_loss
            G_diff = g_square_diff * G_square
            G_output = g_diff_output * G_diff

        elif mode == 'binary':
            y, output = aux
            shape = output.shape
            
            g_loss_entropy = np.ones(shape) / np.prod(shape)
            g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy

        elif mode == 'select':
            output, y, entropy = aux

            g_loss_entropy = 1.0 / np.prod(entropy.shape)
            g_entropy_output = softmax_cross_entropy_with_logits_derv(y, output)

            G_entropy = g_loss_entropy * G_loss
            G_output = g_entropy_output * G_entropy

        return G_output



    # -----------------------------------------------
    # 결과값과 관련된 코드입니다

    def eval_accuracy(self, x, y, output, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            mse = np.mean(np.square(output-y))
            accuracy = 1 - np.sqrt(mse) / np.mean(y)

        elif mode == 'binary':
            estimate = np.greater(output, 0)
            answer = np.equal(y, 1.0)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)

        elif mode == 'select':
            estimate = np.argmax(output, axis=1)
            answer = np.argmax(y, axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)

        return accuracy


    def get_estimate(self, output, mode=None):
        if mode is None: mode = self.mode

        if mode == 'regression':
            estimate = output

        elif mode == 'binary':
            estimate = sigmoid(output)

        elif mode == 'select':
            estimate = softmax(output)

        return estimate

    def train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        print(f'Epoch {epoch} : cost - {np.mean(costs):.3f}  accuracy - {np.mean(accs):.3f}/{acc:.3f}   {time1}/{time2} secs')
    
    def test_prt_result(self, name, acc, time):
        print(f'Result of {name} : accuracy - {acc:.3f}   {time} secs')
    

    
    