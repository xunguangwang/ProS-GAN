import os
import torch
import torch.optim as optim
from torch.autograd import Variable

from model.backbone import *
from utils.hamming_matching import *
from model.utils import *


class DPH(object):
    def __init__(self, args):
        super(DPH, self).__init__()
        self.bit = args.bit
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.backbone = args.backbone
        self.model_name = '{}_DPH_{}_{}'.format(args.dataset, self.backbone, args.bit)
        print(self.model_name)
        self.args = args

        self._build_graph()

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            self.model = AlexNet(self.args.bit)
        elif 'VGG' in self.backbone:
            self.model = VGG(self.backbone, self.args.bit)
        else:
            self.model = ResNet(self.backbone, self.args.bit)
        self.model = self.model.cuda()

    def load_model(self):
        self.model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth'))
        self.model = self.model.cuda()
    
    def CalcSim(self, batch_label, train_label):
        S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
        S = 2 * S - 1
        return S

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1**(epoch // (self.args.n_epochs // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def generate_code(self, data_loader, num_data):
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter, data in enumerate(data_loader, 0):
            data_input, _, data_ind = data
            data_input = Variable(data_input.cuda())
            output = self.model(data_input)
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        return B

    def train(self, train_loader, train_labels, num_train):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)

        U = torch.zeros(num_train, self.bit)
        train_labels = train_labels.cuda()
        for epoch in range(self.args.n_epochs):
            epoch_loss = 0.0
            # training epoch
            for iter, traindata in enumerate(train_loader, 0):
                train_input, train_label, batch_ind = traindata
                train_label = torch.squeeze(train_label)

                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
                S = self.CalcSim(train_label, train_labels)

                self.model.zero_grad()
                train_outputs = self.model(train_input)
                batch_size_ = train_label.size(0)
                
                for i, ind in enumerate(batch_ind):
                    U[ind, :] = train_outputs.data[i]

                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / self.bit
                logloss = (theta_x - S.cuda()) ** 2
                loss = logloss.sum() / (num_train * len(train_label))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print('Epoch: %3d/%3d\tTrain_loss: %3.5f' %
                  (epoch + 1, self.args.n_epochs, epoch_loss / len(train_loader)))
            optimizer = self.adjust_learning_rate(optimizer, epoch)

        torch.save(self.model,
                   os.path.join(self.args.save, self.model_name + '.pth'))

    def test(self, database_loader, test_loader, database_labels, test_labels,
             num_database, num_test):
        self.model.eval()
        qB = self.generate_code(test_loader, num_test)
        dB = self.generate_code(database_loader, num_database)
        map_ = CalcMap(qB, dB, test_labels.numpy(), database_labels.numpy())
        print('Test_MAP(retrieval database): %3.5f' % (map_))
