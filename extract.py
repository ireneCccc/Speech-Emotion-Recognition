import os, matlab.engine, ser
m = matlab.engine.start_matlab()
parent_dir = os.path.dirname(os.getcwd())
dataset = ['TESS', 'RAVDESS']
for dataset_no in range(0, 2):
    dataset_dir = os.path.join(parent_dir, '', dataset[dataset_no])
    for i in range(1, 8):
        print ('Reading emotion #' + str(i) + ' in ' + dataset[dataset_no] + '...', end=' ')
        emotion_dir = os.path.join(dataset_dir, '', str(i))
        file_count = 0
        for file in os.listdir(emotion_dir):
            if file.endswith('.wav'):
                x, fs = ser.audioread(os.path.join(emotion_dir, '', file))
                s, f = ser.spectrogram(x, fs)
                if 'fb' not in dir():
                    fb = ser.mfcc_fb(f) # need only to compute filter once
                mfcc = ser.mfcc(s, fb)
                file_count += 1
                break
        print (str(file_count) + ' files feature extracted')
        break
    #
