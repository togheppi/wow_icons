import torch
import torchvision.utils as vutils
import models
import utils
import numpy as np
import os

# Model parameters
model_name = 'animeGAN3'
image_size = 64
tag_dims = [10]
n_tags = int(np.sum(tag_dims))

G_output_dim = 3

dataset = 'wow_icons_tag'
class_type = 'armor'
subfolder = 'train_' + class_type

# Directories
save_dir = dataset + '/' + model_name + '_result/'
model_dir = dataset + '/' + model_name + '_model/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Models
upsample = 'deconv'
if 'anime' in model_name:
    G_input_dim = 100
    ngf = 64
    G = models.animeGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, upsample=upsample)
elif 'Chainer' in model_name:
    G_input_dim = 128
    ngf = 1024
    G = models.ChainerDRAGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, upsample=upsample, norm='batch')
elif model_name == 'MoeGAN2':
    G_input_dim = 128
    ngf = 64
    num_resnet = 16
    G = models.MoeGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, num_resnet, upsample=upsample, norm='batch')


epoch = 100
G.cuda()
G.load_state_dict(torch.load(model_dir + 'generator_%s_param_epoch_%d.pkl' % (upsample, epoch)))


# Image for test
nrow = n_tags
num_test_samples = nrow * nrow
noise_batch = torch.randn(num_test_samples, G_input_dim).view(-1, G_input_dim, 1, 1)
# Tag for test
all_tag = torch.zeros(num_test_samples, n_tags)
for i in range(n_tags):
    all_tag[i*n_tags:(i+1)*n_tags, i] = 1

# Tag for test
fixed_tag1 = torch.zeros(num_test_samples, n_tags)
for i in range(num_test_samples):
    # fixed tag
    rv1 = 5     # pants
    # rv2 = 2     # blonde hair
    # rv3 = 15    # blue eyes
    # one hot vector
    temp = torch.zeros(n_tags, 1)
    for j in range(n_tags):
        if j == rv1:
            temp[j] = 1

    fixed_tag1[i] = temp

# Tag for test
fixed_tag2 = torch.zeros(num_test_samples, n_tags)
for i in range(num_test_samples):
    # fixed tag
    rv1 = 7     # boots
    # rv2 = 4     # black hair
    # rv3 = 16    # red eyes
    # one hot vector
    temp = torch.zeros(n_tags, 1)
    for j in range(n_tags):
        if j == rv1:
            temp[j] = 1

    fixed_tag2[i] = temp

# Tag for test
random_tag = torch.zeros(num_test_samples, n_tags)
for i in range(num_test_samples):
    # random tag
    rv1 = np.random.randint(0, 2)       # short / long hair
    rv2 = np.random.randint(2, 15)      # hair color
    rv3 = np.random.randint(15, 26)     # eyes color

    # one hot vector
    temp = torch.zeros(n_tags, 1)
    for j in range(n_tags):
        if j == rv1 or j == rv2 or j == rv3:
            temp[j] = 1

    random_tag[i] = temp

# Specific Tag for test
hair_color_tag = torch.zeros(num_test_samples, n_tags)
for i in range(num_test_samples):
    # random tag
    rv1 = np.random.randint(0, 2)       # short / long hair
    rv2 = i % 13 + 2                         # hair colors
    rv3 = np.random.randint(15, 26)     # eyes color

    # one hot vector
    temp = torch.zeros(n_tags, 1)
    for j in range(n_tags):
        if j == rv1 or j == rv2 or j == rv3:
            temp[j] = 1

    hair_color_tag[i] = temp

# generate fake images with fixed noise
G.eval()
fake_batch1 = G(utils.to_var(noise_batch), utils.to_var(all_tag))
fake_batch2 = G(utils.to_var(noise_batch), utils.to_var(fixed_tag1))
fake_batch3 = G(utils.to_var(noise_batch), utils.to_var(fixed_tag2))
vutils.save_image(fake_batch1.data, '%s/%s_random_noise_all_tag.png' % (save_dir, model_name), nrow=nrow, normalize=True, range=(-1.0, 1.0))
vutils.save_image(fake_batch2.data, '%s/%s_random_noise_fixed_tag1.png' % (save_dir, model_name), nrow=nrow, normalize=True, range=(-1.0, 1.0))
vutils.save_image(fake_batch3.data, '%s/%s_random_noise_fixed_tag2.png' % (save_dir, model_name), nrow=nrow, normalize=True, range=(-1.0, 1.0))
