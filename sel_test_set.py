import os, random
parent_dir = os.path.dirname(os.getcwd())

for dataset in ['TESS', 'RAVDESS']:
    dataset_dir = os.path.join(parent_dir, dataset)
    for emotion in range(1, 8):
        print('Selecting emotion #' + str(emotion) + ' in ' + dataset + '...')
        emotion_dir = os.path.join(dataset_dir, str(emotion))
        emotion_dest = os.path.join(parent_dir, dataset + '_test', str(emotion))
        filelist = [file for file in os.listdir(emotion_dir) if file.endswith('.wav') and file[0] != '.']
        random.shuffle(filelist)
        if not os.path.exists(emotion_dest):
            os.makedirs(emotion_dest)
        for file in filelist[:int(0.15 * len(filelist))]:
            os.rename(os.path.join(emotion_dir, file), os.path.join(emotion_dest, file))
print('Finished')
