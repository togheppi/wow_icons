import os
import shutil

data_dir = '../Data/wow_icons/'
save_dir = './data/wow_icons/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class_name_list = ['axe', 'bag', 'belt', 'boot', 'bow', 'bracer', 'cape', 'chest', 'fish', 'food', 'gem', 'gauntlet', 'glove',
                   'hammer', 'helm', 'herb', 'knife', 'mace', 'necklace', 'pant', 'potion', 'rifle', 'ring', 'shield', 'shortblade',
                   'shoulder', 'staff', 'sword']

for class_name in class_name_list:
    print('organizing %s class...' % class_name)
    class_dir = save_dir + class_name
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    filelist = os.listdir(data_dir)
    cnt = 0
    for file in filelist:
        extension = ''.join(os.path.splitext(file)[1])
        name = ''.join(os.path.splitext(file)[0])
        ext = extension.strip('.')

        name = name.lower()
        if class_name in name:
            cnt += 1
            shutil.move(data_dir + file, class_dir + '/' + '%04d.png' % cnt)

            print('%d files are moved to %s.' % (cnt, class_name))



