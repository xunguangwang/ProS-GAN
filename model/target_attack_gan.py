import os
import time
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable

from model.module import * 
from model.utils import *
from utils.hamming_matching import *


class TargetAttackGAN(nn.Module):
    def __init__(self, args):
        super(TargetAttackGAN, self).__init__()
        self.bit = args.bit
        classes_dic = {'FLICKR-25K': 38, 'NUS-WIDE':21, 'MS-COCO': 80}
        rec_weight_dic = {'FLICKR-25K': 100, 'NUS-WIDE':50, 'MS-COCO': 50}
        self.num_classes = classes_dic[args.dataset]
        self.rec_w = rec_weight_dic[args.dataset]
        self.dis_w = 1
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
        self.lr = args.lr
        self.args = args

        self._build_model()

    def _build_model(self):
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(
            Discriminator(num_classes=self.num_classes)).cuda()
        self.prototype_net = nn.DataParallel(PrototypeNet(self.bit, self.num_classes)).cuda()
        self.hashing_model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth')).cuda()
        self.hashing_model.eval()

        # self.criterionGAN = GANLoss(self.args.gan_mode).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_hash_code(self, data_loader, num_data):
        B = torch.zeros(num_data, self.bit)
        self.train_labels = torch.zeros(num_data, self.num_classes)
        for it, data in enumerate(data_loader, 0):
            data_input = data[0]
            data_input = Variable(data_input.cuda())
            output = self.hashing_model(data_input)

            batch_size_ = output.size(0)
            u_ind = np.linspace(it * self.batch_size,
                                np.min((num_data,
                                        (it + 1) * self.batch_size)) - 1,
                                batch_size_,
                                dtype=int)
            B[u_ind, :] = torch.sign(output.cpu().data)
            self.train_labels[u_ind, :] = data[1]
        return B

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        # old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def test_prototype(self, target_labels, database_loader, database_labels, num_database, num_test):
        targeted_labels = np.zeros([num_test, self.num_classes])
        qB = np.zeros([num_test, self.bit])
        for i in range(num_test):
            select_index = np.random.choice(range(target_labels.size(0)), size=1)
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
            targeted_labels[i, :] = batch_target_label.numpy()[0]

            _, target_hash_l, __ = self.prototype_net(batch_target_label.cuda().float())
            qB[i, :] = torch.sign(target_hash_l.cpu().data).numpy()[0]

        database_code_path = os.path.join('log', 'database_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)
        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))

    def train(self, train_loader, target_labels, train_labels, database_loader, database_labels, num_database, num_train, num_test):
        # L2 loss function
        criterion_l2 = torch.nn.MSELoss()

        # Optimizers
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]

        # prototype net
        if os.path.exists(os.path.join(self.args.save, 'prototypenet_{}.pth'.format(self.model_name))):
            self.load_prototypenet()
        else:
            self.train_prototype_net(train_loader, target_labels, num_train)
        self.prototype_net.eval()
        self.test_prototype(target_labels, database_loader, database_labels, num_database, num_test)

        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.lr))
            for i, data in enumerate(train_loader):
                real_input, batch_label, batch_ind = data
                real_input = set_input_images(real_input)
                batch_label = batch_label.cuda()

                select_index = np.random.choice(range(target_labels.size(0)), size=batch_label.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                feature, target_hash_l, _ = self.prototype_net(batch_target_label)
                target_hash_l = torch.sign(target_hash_l.detach())
                fake_g, _ = self.generator(real_input, feature.detach())

                # update D
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    real_d = self.discriminator(real_input)
                    # stop backprop to the generator by detaching
                    fake_d = self.discriminator(fake_g.detach())
                    real_d_loss = self.criterionGAN(real_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(fake_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()

                # update G
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()

                fake_g_d = self.discriminator(fake_g)
                fake_g_loss = self.criterionGAN(fake_g_d, batch_target_label, True)
                reconstruction_loss = criterion_l2(fake_g, real_input)

                target_hashing_g = self.hashing_model((fake_g + 1) / 2)
                logloss = target_hashing_g * target_hash_l
                logloss = torch.mean(logloss)
                logloss = (-logloss + 1)

                # backpropagation
                g_loss = self.rec_w * reconstruction_loss + 5*logloss + self.dis_w*fake_g_loss
                g_loss.backward()
                optimizer_g.step()

                if i % self.args.sample_freq == 0:
                    self.sample(fake_g, '{}/{}/'.format(self.args.sample, self.model_name), str(epoch) + '_' + str(i) + '_fake')
                    self.sample(real_input, '{}/{}/'.format(self.args.sample, self.model_name), str(epoch) + '_' + str(i) + '_real')

                if i % self.args.print_freq == 0:
                    print('step: {:3d} g_loss: {:.3f} d_loss: {:.3f} hash_loss: {:.3f} r_loss: {:.7f}'
                        .format(i, fake_g_loss, d_loss, logloss, reconstruction_loss))

            self.update_learning_rate()

        self.save_generator()

    def train_prototype_net(self, train_loader, target_labels, num_train):
        optimizer_l = torch.optim.Adam(self.prototype_net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = self.args.n_epochs * 2
        epochs = 1
        steps = 300
        batch_size = 64
        lr_steps = epochs * steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_l, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()

        # hash codes of training set
        B = self.generate_hash_code(train_loader, num_train)
        B = B.cuda()

        for epoch in range(epochs):
            for i in range(steps):
                select_index = np.random.choice(range(target_labels.size(0)), size=batch_size)
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                optimizer_l.zero_grad()
                S = CalcSim(batch_target_label.cpu(), self.train_labels)
                _, target_hash_l, label_pred = self.prototype_net(batch_target_label)
                theta_x = target_hash_l.mm(Variable(B).t()) / 2
                logloss = (Variable(S.cuda()) * theta_x - log_trick(theta_x)).sum() / (num_train * batch_size)
                logloss = -logloss
                regterm = (torch.sign(target_hash_l) - target_hash_l).pow(2).sum() / (1e4 * batch_size)
                classifer_loss = criterion_l2(label_pred, batch_target_label)
                loss = logloss + classifer_loss + regterm
                loss.backward()
                optimizer_l.step()
                if i % self.args.sample_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, logloss:{:.5f}, regterm: {:.5f}, l2_loss: {:.7f}'
                        .format(epoch, i, scheduler.get_last_lr()[0], logloss, regterm, classifer_loss))
                scheduler.step()

        self.save_prototypenet()

    def save_prototypenet(self):
        torch.save(self.prototype_net.module.state_dict(),
            os.path.join(self.args.save, 'prototypenet_{}.pth'.format(self.model_name)))

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
            os.path.join(self.args.save, 'generator_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.dis_w)))

    def load_prototypenet(self):
        self.prototype_net.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'prototypenet_{}.pth'.format(self.model_name))))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'generator_{}_{}_{}.pth'.format(self.model_name, self.rec_w, self.dis_w))))

    def load_model(self):
        self.load_prototypenet()
        self.load_generator()

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    def cross_network_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        self.hashing_model.eval()
        self.prototype_net.eval()
        self.generator.eval()
        qB = np.zeros([num_test, self.bit])
        targeted_labels = np.zeros([num_test, self.num_classes])

        perceptibility = 0
        start = time.time()
        for it, data in enumerate(test_loader):
            data_input, data_label, data_ind = data

            select_index = np.random.choice(range(target_labels.size(0)),
                                            size=data_ind.size(0))
            batch_target_label = target_labels.index_select(
                0, torch.from_numpy(select_index))
            targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()

            data_input = set_input_images(data_input)
            feature = self.prototype_net(batch_target_label.cuda())[0]
            target_fake, mix_image = self.generator(data_input, feature)
            target_fake = (target_fake + 1) / 2
            data_input = (data_input + 1) / 2 

            perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)

            self.sample(target_fake, 'result/{}/{}/'.format(self.args.sample, self.model_name), str(it)+'_fake')
            self.sample(data_input, 'result/{}/{}/'.format(self.args.sample, self.model_name), str(it)+'_real')

            target_hashing = self.hashing_model(target_fake)
            qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()

        end = time.time()
        print('Running time: %s Seconds'%(end-start))
        np.savetxt(os.path.join('log', 'test_code_{}_gan_{}.txt'.format(self.args.dataset, self.bit)), qB, fmt="%d")
        np.savetxt(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit)), targeted_labels, fmt="%d")
        database_code_path = os.path.join('log', 'database_code_{}.txt'.format(self.args.target_model.split('.')[0]))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)
        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))
        map_ = CalcMap(qB, dB, test_labels, database_labels.numpy())
        print('Test_MAP(retrieval database): %3.5f' % (map_))
    
    def test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        self.prototype_net.eval()
        self.generator.eval()
        qB = np.zeros([num_test, self.bit])
        targeted_labels = np.zeros([num_test, self.num_classes])

        perceptibility = 0
        start = time.time()
        for it, data in enumerate(test_loader):
            data_input, _, data_ind = data

            select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
            targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()

            data_input = set_input_images(data_input)
            feature = self.prototype_net(batch_target_label.cuda())[0]
            target_fake, mix_image = self.generator(data_input, feature)
            target_fake = (target_fake + 1) / 2
            data_input = (data_input + 1) / 2 

            target_hashing = self.hashing_model(target_fake)
            qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()

            perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)

        end = time.time()
        print('Running time: %s Seconds'%(end-start))
        np.savetxt(os.path.join('log', 'test_code_{}_gan_{}.txt'.format(self.args.dataset, self.bit)), qB, fmt="%d")
        np.savetxt(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit)), targeted_labels, fmt="%d")
        database_code_path = os.path.join('log', 'database_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=np.float)
        else:
            dB = self.generate_hash_code(database_loader, num_database)
            dB = dB.numpy()
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))
        map_ = CalcMap(qB, dB, test_labels, database_labels.numpy())
        print('Test_MAP(retrieval database): %3.5f' % (map_))

    def transfer_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test, target_model_path):
        self.hashing_model = torch.load(os.path.join(self.args.save, target_model_path))
        self.bit = self.args.t_bit
        self.cross_network_test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
