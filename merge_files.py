import os
from PIL import Image
import pickle
import numpy as np

data_dir = '../Data/wow_icons_tag/'
class_type = 'all'
save_dir = '../Data/wow_icons_tag/train' + '_' + class_type

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class_name_list = ['axe', 'bag', 'belt', 'boot', 'bow', 'bracer', 'cape', 'chest', 'fish', 'food', 'gem', 'gauntlet', 'glove',
                   'hammer', 'helm', 'herb', 'knife', 'mace', 'necklace', 'pant', 'potion', 'rifle', 'ring', 'shield', 'shortblade',
                   'shoulder', 'staff', 'sword']

# class_name_list = ['helm', 'chest', 'shoulder', 'bracer', 'glove', 'pant', 'belt', 'boot', 'ring', 'necklace']
# class_name_list = ['bag', 'fish', 'gem', 'herb', 'potion']


cnt = 0
label = []
for idx, class_name in enumerate(class_name_list):
    print('organizing %s class...' % class_name)
    class_dir = os.path.join(data_dir, class_name)

    one_hot = [0 for i in range(len(class_name_list))]
    one_hot[idx] = 1

    filelist = os.listdir(class_dir)
    for file in filelist:
        cnt += 1
        img_fn = os.path.join(class_dir, file)
        img = Image.open(img_fn)
        save_fn = os.path.join(save_dir, "%04d.png" % cnt)
        img.save(save_fn)

        label.append(one_hot)

    print('%d files are processed.' % cnt)

tag_fn = save_dir + '/' + class_type + '_label.pkl'
with open(tag_fn, 'wb') as fp:
    pickle.dump(np.array(label), fp)

print('end')

