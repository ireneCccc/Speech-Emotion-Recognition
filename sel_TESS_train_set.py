import os, random

dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'TESS')
for emotion in range(1, 8):
    print('Selecting emotion #' + str(emotion) + '...')
    emotion_dir = os.path.join(dataset_dir, str(emotion))
    emotion_dest = os.path.join(dataset_dir + '_train', str(emotion))
    filelist = [file for file in os.listdir(emotion_dir) if file.endswith('.wav') and file[0] != '.']
    random.shuffle(filelist)
    if not os.path.exists(emotion_dest):
        os.makedirs(emotion_dest)
    for file in filelist[:int(0.4 * len(filelist))]:
        os.rename(os.path.join(emotion_dir, file), os.path.join(emotion_dest, file))
print('Finished')
