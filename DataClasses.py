import os

import numpy as np

from Dataset import Dataset
from MlpModel import MlpModel
from mathUtils import *

class AbaloneDataset(Dataset):
    def __init__(self):
        super(AbaloneDataset, self).__init__('abalone', 'regression')

        rows, _ = load_csv('./data/abalone.csv')

        xs = np.zeros([len(rows),10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == 'I': xs[n, 0] = 1
            if row[0] == 'M': xs[n, 1] = 1
            if row[0] == 'F': xs[n, 2] = 1
            xs[n, 3:] = row[1:-1]
            ys[n, :] = row[-1:]

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answer):
        for n in range(len(xs)):
            x, est, ans, = xs[n], estimates[n], answer[n]
            xstr = vector_to_str(x, '%4.2f')
            print(f'{xstr} => 추정:{est[0]} | 정답:{ans[0]}')    


class PulsarDataset(Dataset):
    def __init__(self):
        super(PulsarDataset, self).__init__('pulsar', 'binary')
    
        rows, _ = load_csv('./data/pulsar.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-1], data[:,-1:], 0.8)
        self.target_names = ['별', '펄서']
        
    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, '%5.1f', 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = 'O'
            if estr != astr: rstr = 'X'
            print('{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}'. \
                  format(xstr, estr, est[0], astr, rstr))


class SteelDataset(Dataset):
    def __init__(self):
        super(SteelDataset, self).__init__('steel', 'select')
    
        rows, headers = load_csv('./data/steel.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-7], data[:,-7:], 0.8)
        
        self.target_names = headers[-7:]
        
    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)


class PulsarSelectDataset(Dataset):
    def __init__(self):
        super(PulsarSelectDataset, self).__init__('pulsarselect', 'select')
    
        rows, _ = load_csv('./data//pulsar.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:,:-1], onehot(data[:,-1], 2), 0.8)
        self.target_names = ['별', '펄서']
        
    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)


class FlowerDataset(Dataset):
    def __init__(self, resolution=[100,100], input_shape=[-1]):
        super(FlowerDataset, self).__init__('flowers', 'select')
        
        path = './data/flowers'
        
        self.target_names = list_dir(path)
        
        images = []
        idxs =[]

        for dx, dname in enumerate(self.target_names):
            subpath = path + '/' + dname
            filenames = list_dir(subpath)
            for fname in filenames:
                if fname[-4:].lower() != '.jpg':
                    continue
                
                imagepath = os.path .join(subpath, fname)
                pixels = load_image_pixels(imagepath, resolution, input_shape)
                images.append(pixels)
                idxs.append(dx)
        # [3]은 결과값의 크기를 맞춰주기 위한 크기 벡터 [100,100,3] 이 들어갈 자리
        self.image_shape = resolution + [3]

        xs = np.asarray(images, np.float32)
        ys = onehot(idxs, len(self.target_names))
        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        draw_images_horz(xs, self.image_shape)
        show_select_results(estimates, answers, self.target_names)


# ad = AbaloneDataset()
# am = MlpModel('abalone_model', ad, [3])
# am.exec_all(epoch_count=10, report=2)


# pd = PulsarDataset()
# pm = MlpModel('pulsar', pd, [4])
# pm.exec_all()
# pm.visualize(6)


# sd = SteelDataset()
# sm = MlpModel('steel', sd, [12,7, 4])
# sm.exec_all(epoch_count=50, report=10, learning_rate=0.00001)


fd = FlowerDataset()
fm = MlpModel('flower', fd, [50,30,10])
fm.exec_all(epoch_count=80, report=4)

