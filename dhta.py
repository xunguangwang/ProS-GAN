import os
import torch
import time
import collections
import pandas as pd
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.data_provider import *
from utils.hamming_matching import *


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def load_model(path):
    model = torch.load(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def target_adv_loss(noisy_output, target_hash):
    loss = -torch.mean(noisy_output * target_hash)
    return loss


def get_alpha(n):
    if n < 1000:
        return 0.1
    elif n >= 1000 and n < 1200:
        return 0.2
    elif n >= 1200 and n < 1400:
        return 0.3
    elif n >= 1400 and n < 1600:
        return 0.5
    elif n >= 1600 and n < 1800:
        return 0.7
    else:
        return 1


def target_hash_adv(model, query, target_hash, epsilon, step=1, iteration=2000, randomize=False):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        alpha = get_alpha(i)
        noisy_output = model(query + delta, alpha)
        loss = target_adv_loss(noisy_output, target_hash)
        loss.backward()

        delta.data = delta - step * delta.grad.detach()
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()


def load_label(filename, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label)


def GenerateCode(model, data_loader, num_data, bit, use_gpu=True):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else:
            data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B


def generate_hash(model, samples, num_data, bit):
    output = model(samples)
    B = torch.sign(output.cpu().data).numpy()
    return B


def hash_anchor_code(hash_codes):
    return torch.sign(torch.sum(hash_codes, dim=0))


def sample_image(image, name, sample_dir='sample/dhta'):
    image = image.cpu().detach()[0]
    image = transforms.ToPILImage()(image.float())
    image.save(os.path.join(sample_dir, name + '.png'), quality=100)


dataset = 'NUS-WIDE'
DATA_DIR = './data/{}'.format(dataset)
DATABASE_FILE = 'database_img.txt'
TEST_FILE = 'test_img.txt'
DATABASE_LABEL = 'database_label.txt'
TEST_LABEL = 'test_label.txt'

epsilon = 8
epsilon = epsilon / 255.
n_t = 9
iteration = 1
method = 'DHTA'
if n_t == 1:
    method = 'P2P'
transfer = False

bit = 32
batch_size = 32
model_name = 'DPSH'
backbone = 'VGG11'

model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
model = load_model(model_path)
database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, model_name, backbone, bit)

if transfer:
    t_model_name = 'DPSH'
    t_bit = 32
    t_backbone = 'VGG11'
    t_model_path = 'checkpoint/{}_{}_{}_{}.pth'.format(dataset, t_model_name, t_backbone, t_bit)
    t_model = load_model(t_model_path)
else:
    t_model_name = model_name
    t_bit = bit
    t_backbone = backbone
t_database_code_path = 'log/database_code_{}_{}_{}_{}.txt'.format(dataset, t_model_name, t_backbone, t_bit)
target_label_path = 'log/target_label_DHTA_{}.txt'.format(dataset)
test_code_path = 'log/test_code_{}_{}_{}.txt'.format(dataset, method, t_bit)


# data processing
dset_database = HashingDataset(DATA_DIR, DATABASE_FILE, DATABASE_LABEL)
dset_test = HashingDataset(DATA_DIR, TEST_FILE, TEST_LABEL)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
num_database, num_test = len(dset_database), len(dset_test)

if os.path.exists(database_code_path):
    database_hash = np.loadtxt(database_code_path, dtype=np.float)
else:
    database_hash = GenerateCode(model, database_loader, num_database, bit)
    np.savetxt(database_code_path, database_hash, fmt="%d")
if os.path.exists(t_database_code_path):
    t_database_hash = np.loadtxt(t_database_code_path, dtype=np.float)
else:
    t_database_hash = GenerateCode(t_model, database_loader, num_database, t_bit)
    np.savetxt(t_database_code_path, t_database_hash, fmt="%d")
print('database hash codes prepared!')

test_labels_int = np.loadtxt(os.path.join(DATA_DIR, TEST_LABEL), dtype=int)
database_labels_int = np.loadtxt(os.path.join(DATA_DIR, DATABASE_LABEL), dtype=int)
test_labels_str = [''.join(label) for label in test_labels_int.astype(str)]
database_labels_str = [''.join(label) for label in database_labels_int.astype(str)]
test_labels_str = np.array(test_labels_str, dtype=str)
database_labels_str = np.array(database_labels_str, dtype=str)


if os.path.exists(target_label_path):
    target_labels = np.loadtxt(target_label_path, dtype=np.int)
else:
    candidate_labels_count = collections.Counter(database_labels_str)
    candidate_labels_count = pd.DataFrame.from_dict(candidate_labels_count, orient='index').reset_index()
    candidate_labels = candidate_labels_count[candidate_labels_count[0] > n_t]['index']
    candidate_labels = np.array(candidate_labels, dtype=str)

    target_labels = []
    for i in range(num_test):
        target_label_str = np.random.choice(candidate_labels)
        target_label = list(target_label_str)
        target_label = np.array(target_label, dtype=int)
        target_labels.append(target_label)

    target_labels = np.array(target_labels, dtype=np.int)
    np.savetxt(target_label_path, target_labels, fmt="%d")


target_labels_str = [''.join(label) for label in target_labels.astype(str)]
qB = np.zeros([num_test, t_bit], dtype=np.float32)
query_anchor_codes = np.zeros((num_test, bit), dtype=np.float)
perceptibility = 0
start = time.time()
for it, data in enumerate(test_loader):
    queries, _, index = data
    
    n = index[-1].item() + 1
    print(n)
    queries = queries.cuda()
    batch_size_ = index.size(0)

    anchor_codes = torch.zeros((batch_size_, bit), dtype=torch.float)
    for i in range(batch_size_):
        target_label_str = target_labels_str[index[0] + i]
        anchor_indexes = np.where(database_labels_str == target_label_str)
        anchor_indexes = np.random.choice(anchor_indexes[0], size=n_t)

        anchor_code = hash_anchor_code(
            torch.from_numpy(database_hash[anchor_indexes]))
        anchor_code = anchor_code.view(1, bit)
        anchor_codes[i, :] = anchor_code

    query_anchor_codes[it*batch_size:it*batch_size+batch_size_] = anchor_codes.numpy()
    query_adv = target_hash_adv(model, queries, anchor_codes.cuda(), epsilon, iteration=iteration)

    u_ind = np.linspace(it * batch_size, np.min((num_test, (it + 1) * batch_size)) - 1, batch_size_, dtype=int)
    if transfer:
        query_code = generate_hash(t_model, query_adv, batch_size_, t_bit)
    else:
        query_code = generate_hash(model, query_adv, batch_size_, bit)
    qB[u_ind, :] = query_code

    perceptibility += F.mse_loss(queries, query_adv).data * batch_size_

end = time.time()


np.savetxt(test_code_path, qB, fmt="%d")
print('Running time: %s Seconds'%(end-start))
print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
a_map = CalcMap(query_anchor_codes, t_database_hash, target_labels, database_labels_int)
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % a_map)
t_map = CalcMap(qB, t_database_hash, target_labels, database_labels_int)
print('[Retrieval Phase] t-MAP(retrieval database): %3.5f' % t_map)
map = CalcMap(qB, t_database_hash, test_labels_int, database_labels_int)
print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
