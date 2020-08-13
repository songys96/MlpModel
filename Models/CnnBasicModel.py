# -*- encoding: utf-8 -*-

import numpy as np

from .AdamModel import AdamModel
from Utils.mathUtils import *

class CnnBasicModel(AdamModel):
        
    def __init__(self, name, dataset, hconfigs, show_maps=False):
        if isinstance(hconfigs, list) and not isinstance(hconfigs[0], (list,int)):
            # hconfig가 한개일 경우 껍질이 벗겨지는 것을 예방
            hconfigs = [hconfigs]
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        
        super(CnnBasicModel, self).__init__(name, dataset, hconfigs)
        self.use_adam = True

    def alloc_layer_param(self, input_shape, hconfig):
        '''
        어떠한 계층을 넣을 것인지 getattr를 통해 찾고
        알맞는 계층(합성곱, 풀링)등을 넣어준다
        '''
        layer_type = get_layer_type(hconfig)

        m_name = 'alloc_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        pm, output_shape = method(input_shape, hconfig)

        return pm, output_shape

    def forward_layer(self, x, hconfig, pm):
        layer_type = get_layer_type(hconfig)

        m_name = 'forward_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        y, aux = method(x, hconfig, pm)

        return y, aux

    def backprop_layer(self, G_y, hconfig, pm, aux):
        layer_type = get_layer_type(hconfig)

        m_name = 'backprop_{}_layer'.format(layer_type)

        method = getattr(self, m_name)
        G_input = method(G_y, hconfig, pm, aux)
        
        return G_input
    
    # -----------------------------------------------------
    # 비선형 활성화 함수 정의하기
    def activate(self, affine, hconfig):
        if hconfig is None: return affine

        func = get_conf_param(hconfig, 'actfunc', 'relu')

        if func == 'none':      return affine
        elif func == 'relu':    return relu(affine)
        elif func == 'sigmoid': return sigmoid(affine)
        elif func == 'tanh':    return tanh(affine)
        else:                   assert 0

    def activate_derv(self, G_y, y, hconfig):
        if hconfig is None: return G_y

        func = get_conf_param(hconfig, 'actfunc', 'relu')

        if func == 'none':      return G_y
        elif func == 'relu':    return relu_derv(y) * G_y
        elif func == 'sigmoid': return sigmoid_derv(y) * G_y
        elif func == 'tanh':    return tanh_derv(y) * G_y
        else:                   assert 0
    
    # -----------------------------------------------------
    # 계층 설정하기 (완전연결계층 합성곱 풀링)

    def alloc_full_layer(self, input_shape, hconfig):
        ''' 완전 연결 계층의 파라미터 설정 '''
        input_cnt = np.prod(input_shape)
        output_cnt = get_conf_param(hconfig, 'width', hconfig)

        weight = np.random.normal(0, self.rand_std, [input_cnt, output_cnt])
        bias = np.zeros([output_cnt])

        return {'w':weight, 'b':bias}, [output_cnt]

    def alloc_conv_layer(self, input_shape, hconfig):
        ''' 합성곱 계층의 파라미터 설정'''

        # input이 3차원 데이터인지 확인
        assert len(input_shape) == 3
    
        xh, xw, xchn = input_shape
        kh, kw = get_conf_param_2d(hconfig, 'ksize')
        ychn = get_conf_param(hconfig, 'chn')

        kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        if self.show_maps: self.kernels.append(kernel)

        return {'k':kernel, 'b':bias}, [xh, xw, ychn]

    def alloc_max_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3

        xh, xw, xchn = input_shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0 
        assert xw % sw == 0

        return {}, [xh//sh, xw//sw, xchn]


    def alloc_avg_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3

        xh, xw, xchn = input_shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0 
        assert xw % sw == 0

        return {}, [xh//sh, xw//sw, xchn]

# ----------------------------------------------------
# 완전 연결계층 관련 메서드 정의

    def forward_full_layer(self, x, hconfig, pm):
        if pm is None: return x, None
        x_org_shape = x.shape

        if len(x.shape) != 2:
            mb_size = x.shape[0]
            x = x.reshape([mb_size, -1])

        affine = np.matmul(x, pm['w']) + pm['b']
        y = self.activate(affine, hconfig)

        return y, [x,y,x_org_shape]

    def backprop_full_layer(self, G_y, hconfig, pm, aux):
        if pm is None: return G_y

        x, y, x_org_shape = aux
        # affine = 일반화한 공간
        G_affine = self.activate_derv(G_y, y, hconfig)

        g_affine_weight = x.transpose()
        g_affine_input = pm['w'].transpose()

        G_weight = np.matmul(g_affine_weight, G_affine)
        G_bias = np.sum(G_affine, axis=0)
        G_input = np.matmul(G_affine, g_affine_input)

        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)

        return G_input.reshape(x_org_shape)

# --------------------------------------------------------
# 합성곱 계층 관련 메서드 정의

    def forward_conv_layer_7loops(self, x, hconfig, pm):
        '''
        이 메서드는 사용되지 않으나 기본 합성곱의 1차원적 판단으로
        각 파라미터별 반복문을 돌아 총 7회 반복수행하게되는 비효율적 코드이다
        '''
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        conv = np.zeros((mb_size, xh, xw, ychn))

        for n in range(mb_size):
            for r in range(xh):
                for c in range(xw):
                    for ym in range(ychn):
                        for i in range(kh):
                            for j in range(kw):
                                rx = r + i - (kh-1) // 2
                                cx = c + j - (kw-1) //2
                                if rx < 0 or rx >= xh: continue
                                if cx < 0 or cx >= xw: continue
                                for xm in range(kchn):
                                    kval = pm['k'][i][j][xm][ym]
                                    ival = x[n][rx][cx][xm]
                                    conv[n][r][c][ym] = kval * ival
        y = self.activate(onv + pm['b'], hconfig)
        return y, [x,y]

    def forward_conv_layer(self, x, hconfig, pm):
        mb_size, xh, xw, xchn = x.shape
        kh, kw, _, ychn = pm['k'].shape

        # --------------------------------------------

        #           should be checked

        # --------------------------------------------
        # x_flat = [mb+xh+xw, kh+kw+xchn]
        x_flat = get_ext_regions_for_conv(x, kh, kw)
        # k_flat = [[kh+kw+xchn], [ychn]]
        k_flat = pm['k'].reshape([kh*kw*xchn, ychn])
        
        # conv_flat = [mb+xh+xw, ychn]
        conv_flat = np.matmul(x_flat, k_flat)
        conv = conv_flat.reshape([mb_size, xh, xw, ychn])

        y = self.activate(conv + pm['b'], hconfig)
        if self.need_maps: self.maps.append(y)

        return y, [x_flat, k_flat, x, y]

    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        x_flat, k_flat, x, y = aux

        kh, kw, xchn, ychn = pm['k'].shape
        mb_size, xh, xw, _ = G_y.shape
        G_conv = self.activate_derv(G_y, y, hconfig)
        G_conv_flat = G_conv.reshape(mb_size*xh*xw, ychn)

        # 여기 확인
        g_conv_k_flat = x_flat.transpose()
        g_conv_x_flat = k_flat.transpose()
        
        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_input = undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

        self.update_param(pm, 'k', G_kernel)
        self.update_param(pm, 'b', G_bias)

        return G_input


# --------------------------------------------------------
# 평균치 풀링 계층 관련 메서드 정의
    
    def forward_avg_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0,1,3,5,2,4)
        x3 = x2.reshape([-1, sh*sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps: self.maps.append(y)

        return y, None

    def backprop_avg_layer(self, G_y, hconfig, pm, aux):
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten() / (sh * sw)

        gx1 = np.zeros([mb_size*yh*yw*chn, sh*sw], dtype='float32')
        for i in range(sh*sw):
            gx1[:, i] = gy_flat
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0,1,4,2,5,3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input


# --------------------------------------------------------
# 최대치 풀링 계층 관련 메서드 정의
    
    def forward_max_layer(self, x, hconfig, pm):
        mb_size, xh, xw, chn = x.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh //sh, xw //sw
        
        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0,1,3,5,2,4)
        x3 = x2.reshape([-1, sh*sw])

        idxs = np.argmax(x3, axis=1)
        y_flat = x3[np.arange(mb_size*yh*yw*chn), idxs]
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps: self.maps.append(y)
        return y, idxs
    
    def backprop_max_layer(self, G_y, hconfig, pm, aux):
        idxs = aux
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten()

        gx1 = np.zeros([mb_size*yh*yw*chn, sh*sw], dtype='float32')
        gx1[np.arange(mb_size*yh*yw*chn), idxs] = gy_flat[:]
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0,1,4,2,5,3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input

    


# --------------------------------------------------------
# 시각화 관련 메서드 정의

    def visualize(self, num):
        print('Model {} Visualization'.format(self.name))

        self.need_maps = self.show_maps
        self.maps = []

        deX, deY = self.dataset.get_visualize_data(num)
        est = self.get_estimate(deX)

        if self.show_maps:
            for kernel in self.kernels:
                kh, kw, xchn, ychn = kernel.shape
                grids = kernel.reshape([kh, kw, -1]).transpose(2,0,1)
                draw_images_horz(grids[0:5, :, :])

            for pmap in self.maps:
                draw_images_horz(pmap[:, :, :, 0])

        self.dataset.visualize(deX, est, deY)

        self.need_maps = False
        self.maps = None






# --------------------------------------------------------
# 레이어 정보 추출 관련 함수


def get_layer_type(hconfig):
    if not isinstance(hconfig, list): return 'full'
    return hconfig[0]

def get_conf_param(hconfig, key, defval=None):
    '''
    계층에서 사용하는 파라미터의 정보를 담는 다
    예. ['full',{'width':8}]
    처음 주어진 계층의 파라미터의 밸류값을 반환하는데 없으면 주어진 defval값을 제공한다.
    '''
    if not isinstance(hconfig, list): return defval
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    return hconfig[1][key]

def get_conf_param_2d(hconfig, key, defval=None):
    '''
    정방형 문제에 대한 효율적 계산
    정방형이미지나 건너뛰기에 사용된다
    '''
    if len(hconfig) <=1: return defval
    if not key in hconfig[1]: return defval
    val = hconfig[1][key]
    if isinstance(val, list):return val
    return [val, val]


# --------------------------------------------------------
# 합성곱 계층 관련 차원 축소 함수

def get_ext_regions_for_conv(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape
    regs = get_ext_regions(x, kh, kw, 0)
    regs = regs.transpose([2,0,1,3,4,5])

    return regs.reshape([mb_size*xh*xw, kh*kw*xchn])

def get_ext_regions(x, kh, kw, fill):
    mb_size, xh, xw, xchn = x.shape

    # 일단 이 두개 이해 안됨
    eh, ew = xh + kh -1, xw + kw - 1
    bh, bw = (kh-1)//2, (kw-1)//2

    x_ext = np.zeros((mb_size, eh, ew, xchn), dtype='float32') + fill
    x_ext[:, bh:bh+xh, bw:bw+xw , :] = x

    regs = np.zeros((xh, xw, mb_size*kh*kw*xchn), dtype='float32')

    for r in range(xh):
        for c in range(xw):
            regs[r,c,:] = x_ext[:,r:r+kh, c:c+kw, :].flatten()

    return regs.reshape([xh, xw, mb_size, kh, kw, xchn])

def undo_ext_regions_for_conv(regs, x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = regs.reshape([mb_size, xh, xw, kh, kw, xchn])
    regs = regs.transpose([1,2,0,3,4,5])

    return undo_ext_regions(regs, kh, kw)

def undo_ext_regions(regs, kh, kw):
    xh, xw, mb_size, kh, kw, xchn = regs.shape

    eh, ew = xh + kh - 1 , xw + xh - 1
    bh, bw = (kh-1)//2, (kw-1)//2

    gx_ext = np.zeros([mb_size, eh, ew, xchn], dtype='float32')

    for r in range(xh):
        for c in range(xw):
            gx_ext[:,r:r+kh, c:c+kw, :] += regs[r,c]

    return gx_ext[:, bh:bh+xh, bw:bw+xw, :]






