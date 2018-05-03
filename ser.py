def audioread(m, filename):
    x, fs = m.audioread(filename, nargout = 2) # all audio files are mono
    if fs != 24414:
        x = resample(x, fs, 24414)
        fs = 24414

def spectrogram(x, win_size = 1024, hop_size = 512):
    return m.spectrogram(x, m.hamming(win_size), win_size - hop_size)

def find_nearest(m, values, ref):
    #diff = 
    #diff = m.repmat(m.transpose(ref), 1, len(values)) - m.repmat(values, len(ref), 1)
    #refs, index = m.min(m.abs(diff))
    return refs, index

def hz2mel(hz):
    return 1127.01028 * m.log(1 + hz / 700)

def mel2hz(mel):
    return (m.exp(mel / 1127.01028) - 1) * 700

def mfcc(spectrogram, min_hz = 80, max_hz = 8000, n_mel = 40, n_dct = 15):
    # fundamental frequency of adult male voice is 85 ~ 180 Hz
    # fundamental frequency of adult female is 165 ~ 255 Hz
    # frequency range of adult male voice is upto 8 kHz
    min_mel = hz2mel(min_hz)
    max_mel = hz2mel(max_hz)
    centers = list(m.linspace(min_mel, max_mel, n_mel)[0]) # linear in mel
    left = 2 * centers[0] - centers[1] # left bottom of the first triangle
    right = 2 * centers[-1] - centers[-2] # right bottom of the last triangle
    centers = [left] + centers + [right]
    centers = list(map(mel2hz, centers)) # convert back to Hz
    
    # create the filterbank matrix
    # number of rows: number of filters
    # number of columns: number of frequency bins in spectrogram, i.e. length of f
    
