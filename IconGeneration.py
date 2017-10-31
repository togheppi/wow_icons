# Automatic character face generation
import torch
from torch.autograd import Variable, grad
import models
from dataset import DatasetFromFolder
import torchvision.transforms as transforms
import os
from logger import Logger
import utils
import numpy as np

# Model parameters
model_name = 'animeGAN_conv2'
image_size = 64
tag_dims = [10]
n_tags = int(np.sum(tag_dims))
G_input_dim = 100
G_output_dim = 3
D_input_dim = 3
D_output_dim = 1

learning_rate = 0.0002
betas = (0.5, 0.999)
batch_size = 64
num_epochs = 100
decay_epoch = 20
data_root = '../Data'
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

# Dataset pre-processing
transform = transforms.Compose([transforms.Scale(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data = DatasetFromFolder(data_root, dataset, class_type, subfolder=subfolder, transform=transform, fliplr=True)
data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Models
upsample_list = ['deconv']
model_len = len(upsample_list)

lamb_adv = 1
lamb_gp = 0.25
lamb_cls = 1/len(tag_dims)

if model_name == 'MoeGAN':
    G_input_dim = 128
    ngf = 64
    ndf = 32
    num_resnet = 16
    G_list = []
    D_list = []
    for upsample in upsample_list:
        G_list.append(models.MoeGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, num_resnet, upsample=upsample, norm='batch'))
        D_list.append(models.MoeGANDiscriminator2(D_input_dim, ndf, D_output_dim, tag_dims, norm='batch'))
elif model_name == 'ChainerDRAGAN':
    G_input_dim = 128
    ngf = 1024
    ndf = 64
    G_list = []
    D_list = []
    for upsample in upsample_list:
        G_list.append(models.ChainerDRAGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, upsample=upsample, norm='batch'))
        D_list.append(models.ChainerDRAGANDiscriminator2(D_input_dim, ndf, D_output_dim, tag_dims, norm='batch'))
elif 'anime' in model_name:
    ngf = 64
    ndf = 64
    G_list = []
    D_list = []
    for upsample in upsample_list:
        G_list.append(models.animeGANGenerator2(G_input_dim, n_tags, ngf, G_output_dim, upsample=upsample, norm='batch'))
        D_list.append(models.animeGANDiscriminator2(D_input_dim, ndf, D_output_dim, tag_dims, norm=None))

# Loss function
BCE_loss = torch.nn.BCELoss()
CE_loss = torch.nn.CrossEntropyLoss()

# Loggers
D_logger = []
G_logger = []
img_logger = []
# Avg. losses
D_avg_losses = []
G_avg_losses = []
# Optimizers
G_optimizer = []
D_optimizer = []

for i in range(model_len):
    # Weight initialize
    G_list[i].weight_init()
    D_list[i].weight_init()
    G_list[i].cuda()
    D_list[i].cuda()

    # Set the logger
    D_log_dir = save_dir + 'D_logs' + '_' + upsample_list[i]
    G_log_dir = save_dir + 'G_logs' + '_' + upsample_list[i]
    if not os.path.exists(D_log_dir):
        os.mkdir(D_log_dir)
    D_logger.append(Logger(D_log_dir))

    if not os.path.exists(G_log_dir):
        os.mkdir(G_log_dir)
    G_logger.append(Logger(G_log_dir))

    img_log_dir = save_dir + 'img_logs' + '_' + upsample_list[i]
    if not os.path.exists(img_log_dir):
        os.mkdir(img_log_dir)
    img_logger.append(Logger(img_log_dir))

    # optimizers
    G_optimizer.append(torch.optim.Adam(G_list[i].parameters(), lr=learning_rate, betas=betas))
    D_optimizer.append(torch.optim.Adam(D_list[i].parameters(), lr=learning_rate, betas=betas))

    # list for avg. losses
    D_avg_losses.append([])
    G_avg_losses.append([])

    # Set discriminator to training mode
    D_list[i].train()

# tag_fn = os.path.join(data_root, dataset) + '/' + dataset + '_label.pkl'
# with open(tag_fn, 'rb') as fp:
#     tags = pickle.load(fp)

# tag = torch.LongTensor(tags).squeeze()

# Fixed noise for test
nrow = 5
num_test_samples = nrow * nrow
fixed_noise = torch.randn(num_test_samples, G_input_dim).view(-1, G_input_dim, 1, 1)

# Tag for test
# all_tag = torch.zeros(num_test_samples, n_tags)
# for i in range(n_tags):
#     all_tag[i*n_tags:(i+1)*n_tags, i] = 1

fixed_tag = torch.zeros(num_test_samples, n_tags)
for i in range(num_test_samples):
    # # random tag
    # rv1 = np.random.randint(0, 2)       # short / long hair
    # rv2 = np.random.randint(2, 15)      # hair color
    # rv3 = np.random.randint(15, 26)     # eyes color

    # fixed tag
    rv1 = 5     # pants
    # rv2 = 2     # blonde hair
    # rv3 = 15    # blue eyes
    # one hot vector
    temp = torch.zeros(n_tags, 1)
    for j in range(n_tags):
        if j == rv1:
            temp[j] = 1

    fixed_tag[i] = temp

# Training GAN
step = 0
for epoch in range(num_epochs):

    D_losses = []
    G_losses = []
    for i in range(model_len):
        D_losses.append([])
        G_losses.append([])

    # learning rate decay
    if (epoch + 1) > decay_epoch:
        for i in range(model_len):
            # D_optimizer[i].param_groups[0]['lr'] = learning_rate * (0.1 ** (epoch + 1 - decay_epoch))
            # G_optimizer[i].param_groups[0]['lr'] = learning_rate * (0.1 ** (epoch + 1 - decay_epoch))
            D_optimizer[i].param_groups[0]['lr'] -= learning_rate / (num_epochs - decay_epoch)
            G_optimizer[i].param_groups[0]['lr'] -= learning_rate / (num_epochs - decay_epoch)
            # lr = learning_rate * np.exp(-0.1*(num_epochs - decay_epoch))
            # D_optimizer[i].param_groups[0]['lr'] = lr
            # G_optimizer[i].param_groups[0]['lr'] = lr
            # lr = learning_rate * np.power(0.5, (epoch+1)//decay_epoch)
            # D_optimizer[i].param_groups[0]['lr'] = lr
            # G_optimizer[i].param_groups[0]['lr'] = lr
        # print('Decaying learning rate: %.g' % lr)

    # minibatch training
    for i in range(model_len):
        G_list[i].train()

    for iter, (images, tags) in enumerate(data_loader):

        # real image data
        mini_batch = images.size()[0]
        real_image = Variable(images.cuda())
        # real tag data
        # real_tag = tag[mini_batch * iter:mini_batch * (iter + 1)]
        # real_tag = torch.FloatTensor(tags)
        real_tag = Variable(tags.cuda())

        # fake noise data
        noise = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
        noise = Variable(noise.cuda())

        # real & fake labels
        real_label = Variable(torch.ones(mini_batch).cuda())
        fake_label = Variable(torch.zeros(mini_batch).cuda())

        for i in range(model_len):
            D_optimizer[i].zero_grad()

            # Train discriminator with real data
            D_real_decision, Tag_real_decision = D_list[i](real_image)
            D_real_loss = BCE_loss(D_real_decision, real_label)

            Tag_real_decision1 = Tag_real_decision
            Tag_real_loss1 = CE_loss(Tag_real_decision1, torch.max(real_tag[:, 0:tag_dims[0]], 1)[1])
            # Tag_real_loss2 = CE_loss(Tag_real_decision2, torch.max(real_tag[:, tag_dims[0]:tag_dims[0]+tag_dims[1]], 1)[1])
            # Tag_real_loss3 = CE_loss(Tag_real_decision3, torch.max(real_tag[:, tag_dims[0]+tag_dims[1]:], 1)[1])
            Tag_real_loss = Tag_real_loss1

            # Train discriminator with fake data
            fake_image = G_list[i](noise, real_tag)
            D_fake_decision, Tag_fake_decision = D_list[i](fake_image)
            D_fake_loss = BCE_loss(D_fake_decision, fake_label)

            Tag_fake_decision1 = Tag_fake_decision
            Tag_fake_loss1 = CE_loss(Tag_fake_decision1, torch.max(real_tag[:, 0:tag_dims[0]], 1)[1])
            # Tag_fake_loss2 = CE_loss(Tag_fake_decision2, torch.max(real_tag[:, tag_dims[0]:tag_dims[0]+tag_dims[1]], 1)[1])
            # Tag_fake_loss3 = CE_loss(Tag_fake_decision3, torch.max(real_tag[:, tag_dims[0]+tag_dims[1]:], 1)[1])
            Tag_fake_loss = Tag_fake_loss1

            # Loss calculation
            D_adv_loss = D_real_loss + D_fake_loss
            D_tag_loss = Tag_real_loss + Tag_fake_loss

            # Gradient penalty
            if 'DRAGAN' in model_name or 'MoeGAN' in model_name:
                x_ = real_image
                alpha = torch.rand(x_.size()).cuda()
                x_hat = Variable(
                    alpha * x_.data + (1 - alpha) * (x_.data + 0.5 * x_.data.std() * torch.rand(x_.size()).cuda()),
                    requires_grad=True)

                pred_hat, _ = D_list[i](x_hat)
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = lamb_cls * D_tag_loss + lamb_adv * D_adv_loss + lamb_gp * gradient_penalty
            else:
                D_loss = lamb_cls * D_tag_loss + lamb_adv * D_adv_loss

            # Back propagation
            D_loss.backward()
            D_optimizer[i].step()

            # Train generator
            G_optimizer[i].zero_grad()
            fake_image = G_list[i](noise, real_tag)
            D_fake_decision, Tag_fake_decision = D_list[i](fake_image)

            # aa = D_fake_decision.data.cpu()
            # if np.isnan(aa.numpy()[0]):
            #     print('nan detected!!')
            #     break
            # elif torch.min(aa) >= 0. and torch.max(aa) <= 1.:
            G_adv_loss = BCE_loss(D_fake_decision, real_label)
            Tag_fake_decision1 = Tag_fake_decision
            Tag_loss1 = CE_loss(Tag_fake_decision1, torch.max(real_tag[:, 0:tag_dims[0]], 1)[1])
            # Tag_loss2 = CE_loss(Tag_fake_decision2, torch.max(real_tag[:, tag_dims[0]:tag_dims[0] + tag_dims[1]], 1)[1])
            # Tag_loss3 = CE_loss(Tag_fake_decision3, torch.max(real_tag[:, tag_dims[0] + tag_dims[1]:], 1)[1])
            G_tag_loss = Tag_loss1

            # Back propagation
            G_loss = lamb_cls * G_tag_loss + lamb_adv * G_adv_loss
            G_loss.backward()
            G_optimizer[i].step()

            # loss values
            D_losses[i].append(D_loss.data[0])
            G_losses[i].append(G_loss.data[0])

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch + 1, num_epochs, iter + 1, len(data_loader), D_loss.data[0], G_loss.data[0]))

            # ============ TensorBoard logging ============#
            D_logger[i].scalar_summary('losses', D_loss.data[0], step + 1)
            G_logger[i].scalar_summary('losses', G_loss.data[0], step + 1)

        step += 1

    # avg loss values for plot
    for i in range(model_len):
        D_avg_loss = torch.mean(torch.FloatTensor(D_losses[i]))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses[i]))

        D_avg_losses[i].append(D_avg_loss)
        G_avg_losses[i].append(G_avg_loss)

    for i in range(model_len):
        # generate fake images with fixed noise
        G_list[i].eval()
        fake_batch = G_list[i](Variable(fixed_noise.cuda(), volatile=True),
                                Variable(fixed_tag.cuda(), volatile=True))

        # save result images.
        gen_images = utils.result_images(fake_batch.data, epoch, edge=False, model_name=model_name + '_' + upsample_list[i],
                                          save_dir=save_dir, nrow=nrow)

        # log the images
        info = {
            'Generated images': utils.to_np(utils.denorm(fake_batch)).transpose(0, 2, 3, 1),  # convert to BxHxWxC
        }

        for key, images in info.items():
            img_logger[i].image_summary(key, images, epoch + 1)

        # Save trained parameters of model
        torch.save(G_list[i].state_dict(), model_dir + 'generator_' + upsample_list[i] + '_param_epoch_%d.pkl' % (epoch+1))
        torch.save(D_list[i].state_dict(), model_dir + 'discriminator' + upsample_list[i] + '_param_epoch_%d.pkl' % (epoch+1))


avg_losses = []
for i in range(model_len):
    # Plot average losses
    avg_losses.append(D_avg_losses[i])
    avg_losses.append(G_avg_losses[i])
    utils.plot_loss(avg_losses, num_epochs, upsample_list[i], save=True, save_dir=save_dir)

    # Save trained parameters of model
    torch.save(G_list[i].state_dict(), model_dir + 'generator_' + upsample_list[i] + '_param.pkl')
    torch.save(D_list[i].state_dict(), model_dir + 'discriminator' + upsample_list[i] + '_param.pkl')
