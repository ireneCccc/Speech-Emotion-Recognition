import sys
argc = len(sys.argv)
if argc < 2:
    print('Please give enough arguments:\n  part[, emo_read_num=7, file_read_num=-1, win_size=1024, hop_size=512]')
    exit()

import numpy as np
import librosa
from scipy.io import wavfile
import os, time, csv, datetime

part = sys.argv[1]
parameters = [7, -1, 1024, 512, 80, 300, 8000]
parameters[:(argc - 2)] = sys.argv[2:]
[emo_read_num, file_read_num, win_size, hop_size, min_freq, max_fund_freq, max_freq] = [int(x) for x in parameters]

TESS_trim = 0.62
RAVDESS_trim = 0.26
fixed_len = 43195

time_very_start = time.time()
print('Start')

features = []
nowdate = datetime.datetime.now()
savename = 'feat_' + part + '_' + str(win_size) + 'win_' + \
           '[' + str(nowdate.month).zfill(2) + str(nowdate.day).zfill(2) + '-' + \
           str(nowdate.hour).zfill(2) + str(nowdate.minute).zfill(2) + '].csv'

parent_dir = os.path.dirname(os.getcwd())
f = open(os.path.join(parent_dir, savename), 'w', newline='')
print('Will write to ' + savename)
spamwriter = csv.writer(f)

for dataset in ['TESS']:
    dataset_dir = os.path.join(parent_dir, dataset + '_' + part)
    for emotion in range(7):
        emotion=6
        if emotion >= emo_read_num:
            break
        time_start = time.time()
        print('Reading emotion #' + str(emotion) + ' in ' + dataset + '_' + part + '...')
        emotion_dir = os.path.join(dataset_dir, str(emotion))
        file_count = 0
        file_list = os.listdir(emotion_dir)
        for file in file_list:
            print(file)
            if file_read_num != -1 and file_count >= file_read_num:
                break
            if (not file.endswith('.wav')) or file[0] == '.':
                continue
            fs, x = wavfile.read(os.path.join(emotion_dir, file))
            
            if len(x) >= fixed_len:
                if dataset == 'TESS':
                    x = x[int((len(x) - fixed_len) / 2):][:fixed_len]
                else:
                    x = x[-fixed_len:]
            else: # add zeros at end
                x = np.concatenate((x, [0] * (fixed_len - len(x))))

            x = x / 32768 # convert 16-bit PCM to [-1, 1]

            s = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=win_size, hop_length=hop_size)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(s), sr=fs)[1:16]

            rms = librosa.feature.rmse(y=x, frame_length=win_size, hop_length=hop_size)
            zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=win_size, hop_length=hop_size)
            centroid = librosa.feature.spectral_centroid(y=x, sr=fs, n_fft=win_size, hop_length=hop_size)
            
            # pitch
            min_lag = int(fs / max_fund_freq)
            max_lag = int(fs / min_freq)
            L = range(min_lag, max_lag + 1)
            spec = librosa.core.stft(x, n_fft=1024, hop_length=512, win_length=1024)
            dividend = np.transpose([np.real(np.fft.ifft(row)) for row in (np.absolute(spec) ** 2).transpose()])
            divisor = np.transpose([win_size - lag + 1 for lag in L])
            acf = dividend[L] / divisor[:, None]
            i_max = np.argmax(acf, axis=0)
            pitch = fs / (i_max - 1 + min_lag)
            
            if len(set([len(mfcc[0]), len(rms[0]), len(zcr[0]), len(centroid[0]), len(pitch)])) != 1:
                print('  Error: File ' + file + ' has different numbers of windows among different features!')
                continue

            if dataset == 'TESS':
                label = [emotion, 0]
                gender = np.vstack(([0] * len(rms[0]), [1] * len(rms[0])))
            else:
                if int(file[19]) % 2 == 0:
                    gender = np.vstack(([0] * len(rms[0]), [1] * len(rms[0]))) # female
                else:
                    gender = np.vstack(([1] * len(rms[0]), [0] * len(rms[0]))) # male
                if file[10] == '1':
                    label = [emotion, 1] # RAVDESS, normal intensity
                else:
                    label = [emotion, 2] # RAVDESS, strong intensity

            # vertically concatenate features of all windows
            concat = np.vstack((mfcc, rms, zcr, centroid, pitch, gender))
            # then flatten it, and concatenate with label
            features.append(list(concat.flatten()) + label)
            file_count += 1
            if len(features) == 50:
                [spamwriter.writerow(v) for v in features]
                print('Write 50 rows')
                features = []
        print('    ' + str(file_count) + ' files feature extracted. (' + str(int(time.time() - time_start)) + ' s)')
    break

if len(features) > 0:
    [spamwriter.writerow(v) for v in features]
    print('Write ' + str(len(features)) + ' rows')
print('Finished. (' + str(int(time.time() - time_very_start)) + ' s in total)')
