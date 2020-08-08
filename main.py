# -*- coding: utf-8 -*-
import sys
import os

from Datasets.DataClasses import FlowerDataset, Office31Dataset
from Models.AdamModel import AdamModel
from Models.MlpModel import MlpModel

sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))


# fd = FlowerDataset()
# fm = AdamModel('flower', fd, [30,10])
# fm.exec_all(epoch_count=32, batch_size=50, report=4, learning_rate = 0.001)
# mlp = MlpModel('flower', fd, [30,10])
# fm.exec_all(epoch_count=32, batch_size=50, report=4, learning_rate = 0.001)

od = Office31Dataset()

# om1 = MlpModel('office31_model_1', od, [10])
# om1.exec_all(epoch_count=20, report=10)

om2 = AdamModel('office31_model_1', od, [32, 10])
om2.use_adam = True
om2.exec_all(epoch_count=50, report=10, learning_rate = 0.0001)
