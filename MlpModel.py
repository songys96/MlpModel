# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mathUtils import *

class Model(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False # ㅇㅣ 플래그는 학습중에만 True
        if not hasattr(self, 'rand_std'): self.rand_std = 0.030

    def __str__(self):
        return f'{self.name} : {self.dataset}'

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):
        self.train(epoch_count, batch_size, learning_rate, report)
        self.test()
        if show_cnt > 0: self.visualize(show_cnt)


class MlpModel(Model):
        
    def __init__(self, name, dataset, hconfigs):
        super(MlpModel, self).__init__(name, dataset)
        self.init_parameters(hconfigs)
    
    def init_parameters(self, hconfigs):
        """
        alloc_layer_param 함수를 사용하여 
        인풋과 아웃풋 히든레이어에 각각 가중치를 부여하는 함수
        """
        self.hconfigs = hconfigs
        self.pm_hiddens = []
        prev_shape = self.dataset.input_shape

        for hconfig in hconfigs:
            pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
            self.pm_hiddens.append(pm_hidden)

        output_cnt = int(np.prod(self.dataset.output_shape))
        self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)

    def alloc_layer_param(self, input_shape, hconfig):
        """
        alloc_param_pair 함수를 이용하여
        가중치를 부여하고 그 가중치 쌍을 반환
        """
        input_cnt = np.prod(input_shape)
        output_cnt = hconfig

        weight, bias = self.alloc_param_pair([input_cnt, output_cnt])

        return {'w':weight, 'b':bias}, output_cnt

    def alloc_param_pair(self, shape):
        """
        가중치 생성 메서드
        """
        weight = np.random.normal(0, self.rand_std, shape)
        bias = np.zeros([shape[-1]])
        return weight, bias

    def train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        """
        model이 exec_all() 시행하면 실행되는 메서드
        학습의 큰 틀이 있고 세부적인 것은 함수화
        여기서는 하이퍼파라미터, 에포크, 배치, 러닝레이트등을 기준으로 코드를 구현
        report 값에 따라 결과값을 찍어냄
        """
        self.learning_rate = learning_rate
        batch_count = int(self.dataset.train_count / batch_size)
        
        time1 = time2 = int(time.time()) # timestamp

        if report != 0:
            print(f'Model {self.name} train started')

        for epoch in range(epoch_count):
            costs = []
            accs = []
            self.dataset.shuffle_train_data(batch_size*batch_count)
            
            for n in range(batch_count):
                # n번째 훈련데이터 배치를 가져오기
                trX, trY = self.dataset.get_train_data(batch_size, n)
                cost, acc = self.train_step(trX, trY)
                costs.append(cost)
                accs.append(acc)

            if report > 0 and (epoch+1) % report == 0:
                vaX, vaY = self.dataset.get_validate_data(100)
                acc = self.eval_accuracy(vaX, vaY)
                time3 = int(time.time())
                tm1, tm2 = time3 - time2, time3 - time1
                self.dataset.train_prt_result(epoch+1, costs, accs, acc, tm1, tm2)
                time2 = time3

        tm_total = int(time.time()) - time1
        print(f'Model {self.name} train ended in {tm_total} secs')

    def train_step(self, x, y):
        """
        실질적인 학습과정이 일어나는 곳
        순전파, 순전파처리, 역전파전처리, 역전파 순으로 일어난다

        """
        self.is_training = True

        #aux_nn = output의 보조정보
        output, aux_nn = self.forward_neuralnet(x)
        # aux_pp = 손실함수의 보조정보
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(x, y, output)

        G_loss = 1
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        self.is_training = False

        return loss, accuracy

    def forward_neuralnet(self, x):
        """
        순전파 과정을 통해
        이 과정에서 forward_layer()함수를 통해 가중치와 편향이 추가된다
        은닉계층의 output
        출력계층의 output을 구하여 반환
        """
        # hidden을 prev라고 생각하면 좋을것같음
        hidden = x
        aux_layers = []

        for n, hconfig in enumerate(self.hconfigs):
            # hidden = output으로 업데이트 , aux는 [x, output]
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])
            aux_layers.append(aux)
        output, aux_out = self.forward_layer(hidden, None, self.pm_output)

        return output, [aux_out, aux_layers]

    def forward_layer(self, x, hconfig, pm):
        """
        가중치와 편향을 적용시키고 
        은닉계층에서 음의 output은 0으로 변경
        """
        y = np.matmul(x, pm['w']) + pm['b']
        if hconfig is not None:
            y = relu(y)
        return y, [x,y]

    def forward_postproc(self, output, y):
        """
        dataset에 맞는 손실함수를 구하기 위해 
        dataset의 forward_postproc 메서드를 이용

        """
        loss, aux_loss = self.dataset.forward_postproc(output, y)
        extra, aux_extra = self.forward_extra_cost(y)
        return loss+extra, [aux_loss, aux_extra]

    def forward_extra_cost(self, y):
        """
        추후 정규화 장치 도입을 위해 미리 만들어둠
        """
        return 0, None

    def backprop_postproc(self, G_loss, aux):
        """
        dataset에 맞는 역전파 전처리 과정으로
        output변화량에 따른 손실함수의 변화량을 구할 수 있는데
        이전에 dataset의 forward_postproc과정에서 사용한 손실함수(소프트맥스교차엔트로피, mse등등)
        의 미분값을 구하는 과정이다
        """
        aux_loss, aux_extra = aux
        self.backprop_extra_cost(G_loss, aux_extra)
        G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
        return G_output

    def backprop_extra_cost(self, G_loss, aux):
        pass


    def backprop_neuralnet(self, G_output, aux):
        aux_out, aux_layers = aux

        G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)

        for n in reversed(range(len(self.hconfigs))):
            hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)

        return G_hidden

    def backprop_layer(self, G_y, hconfig, pm, aux):
        """
        G_output은 output 변화량에 따른 손실함수의 기울기
        따라서
        G_weight은 가중치에 따른 손실함수의 기울기
            이 값으로 w업데이트
        G_input은 hidden값(이전 결과값)의 변화량에 따른 손실함수 기울기
        이 G_input값이 이전 은닉계층의 G_ouput값으로 해석됨
        """
        x, y = aux
        
        if hconfig is not None:
            G_y = relu_derv(y) * G_y
        
        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)

        pm['w'] -= self.learning_rate * G_weight
        pm['b'] -= self.learning_rate * G_bias

        return G_input

    def test(self):
        teX, teY = self.dataset.get_test_data()
        time1 = int(time.time())
        acc = self.eval_accuracy(teX, teY)
        time2 = int(time.time())
        self.dataset.test_prt_result(self.name, acc, time2-time1)

    def visualize(self, num):
        """
        시각화 구현
        num은 show_cnt와 동일
        dataset에서 시각화 자료를 가져오고
        get_estimate함수에 넣어 추정 정보를 얻고
        dataset의 시각화 메서드에 제공해 출력을 받는다
        dataset에게 다시 제공하는 이유는 데이터셋의 내용에 맞는 구조로 구현하기 위함!
        """
        print(f'Model {self.name} Visualization')
        deX, deY = self.dataset.get_visualize_data(num)
        est = self.get_estimate(deX)
        self.dataset.visualize(deX, est, deY)


    def eval_accuracy(self, x, y, output=None):
        if output is None:
            output, _ = self.forward_neuralnet(x)
        accuracy = self.dataset.eval_accuracy(x, y ,output)
        return accuracy
    
    def get_estimate(self, x):
        output, _ = self.forward_neuralnet(x)
        estimate = self.dataset.get_estimate(output)
        return estimate








