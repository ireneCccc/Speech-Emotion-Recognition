import matlab, ser, os, time, random, csv, datetime

# compute mean, variance, maximum, minimum, mean of first half, mean of second half for each row (feature)
def aggregate(matrix):
    return ser.mat2list(m.mean(matrix)) + \
           ser.mat2list(m.var(matrix)) + \
           ser.mat2list(m.max(matrix)) + \
           ser.mat2list(m.min(matrix)) + \
           ser.mat2list(m.mean(matrix[:int(len(matrix) / 2)])) + \
           ser.mat2list(m.mean(matrix[int(len(matrix) / 2):])) + \
           ser.mat2list(m.var(matrix[:int(len(matrix) / 2)])) + \
           ser.mat2list(m.var(matrix[int(len(matrix) / 2):]))

def run(parent_dir, matlab_engine, arguments, emo_read_num = 7, file_read_num = -1, task = 'train', TESS_trim = 0.62, RAVDESS_trim = 0.26):
    global m
    m = matlab_engine
    ser.get_matlab_engine(m, arguments)
    time_very_start = time.time()
    print('Run with arguments: [' + ', '.join([str(v) for v in arguments]) + ']')
    
    features = []
    nowdate = datetime.datetime.now()
    savename = 'feat_' + str(arguments[0]) + 'win_' + task + \
               '[' + str(nowdate.month).zfill(2) + str(nowdate.day).zfill(2) + '-' + \
               str(nowdate.hour).zfill(2) + str(nowdate.minute).zfill(2) + '].csv'
    f = open(os.path.join(parent_dir, savename), 'w', newline='')
    print('Will write to ' + savename)
    spamwriter = csv.writer(f)

    for dataset in ['TESS', 'RAVDESS']:
        dataset_dir = os.path.join(parent_dir, dataset + '_' + task)
        emotion_list = [v for v in range(1, 8)]
        for emotion in emotion_list[:emo_read_num]:
            time_start = time.time()
            print('Reading emotion #' + str(emotion) + ' in ' + dataset + '_' + task + '...')
            emotion_dir = os.path.join(dataset_dir, str(emotion))
            file_count = 0
            file_list = os.listdir(emotion_dir)
            for file in file_list:
                if file_read_num != -1 and file_count >= file_read_num:
                    break
                if file.endswith('.wav') and file[0] != '.':
                    x, fs = ser.audioread(os.path.join(emotion_dir, file))
                    
                    if dataset == 'TESS': # The first 62% (TESS_trim) of audio in TESS is useless
                        x = x
                        #x = x[int(TESS_trim * len(x)):]
                    else: # The first and last 26% (RAVDESS_trim) of audio in RAVDESS is silence
                        x = x[int(RAVDESS_trim * len(x)):int((1 - RAVDESS_trim) * len(x))]
                    
                    s, f = ser.spectrogram(x, fs)
                    if 'fb' not in dir(): # need only to compute filter once
                        fb = ser.mfcc_fb(f)
                    mfcc = ser.mfcc(s, fb)
                    energy, dE = ser.energy(x, len(s[0]))
                    pitch, dpitch = ser.pitch(s, fs)

                    if len(set([len(mfcc[0]), len(energy[0]), len(pitch[0])])) != 1:
                        print('  Error: File ' + file + ' has different numbers of windows among different features!')
                    else:
                        # vertically concatenate features of all windows (then transpose)
                        concat = m.transpose(m.vertcat(energy, pitch, dE, dpitch, mfcc))
                        
                        # remove windows where energy is very low (= silence)
                        if dataset == 'TESS':
                            threshold = 2e-05
                        else: # RAVDESS
                            threshold = 5e-09
                        concat = matlab.double([y for y in concat if y[0] > threshold]) # energy is placed at row 0

                        label = []
                        if dataset == 'TESS':
                            label.append(1) # female
                            label.append(emotion)
                            label.append(0)
                        else:
                            if int(file[19]) % 2 == 0:
                                label.append(1) # female
                            else:
                                label.append(0) # male
                            label.append(emotion)
                            if file[10] == '1':
                                label.append(1) # RAVDESS, normal intensity
                            else:
                                label.append(2) # RAVDESS, strong intensity
                        features.append(aggregate(concat) + label)
                        
                    file_count += 1
                    if len(features) == 50:
                        [spamwriter.writerow(v) for v in features]
                        print('Write 50 rows')
                        features = []
            print('    ' + str(file_count) + ' files feature extracted. (' + str(int(time.time() - time_start)) + ' s)')
    #m.imagesc(matlab.double(features))
    
    if len(features) > 0:
        [spamwriter.writerow(v) for v in features]
        print('Write ' + str(len(features)) + ' rows')
    print('Finished. (' + str(int(time.time() - time_very_start)) + ' s in total)')
