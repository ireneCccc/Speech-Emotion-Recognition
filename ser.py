global win_size
global hop_size

win_size = 1024.0
hop_size = 512.0

def get_matlab_engine(matlab_module, matlab_engine):
    global matlab
    global m
    matlab = matlab_module
    m = matlab_engine

def audioread(filename):
    x, fs = m.audioread(filename, nargout = 2) # all audio files are mono
    if fs != 24414:
        x = m.resample(x, fs, 24414.0)
        fs = 24414
    return x, fs

def spectrogram(x, fs):
    s, w = m.spectrogram(x, m.hamming(win_size), win_size - hop_size, nargout = 2)
    f = m.times(w, fs / 2.0 / m.pi())
    return s, f

# Find indices of nearest values in the reference list
def find_nearest(val, ref):
    i = 0
    j = 0
    res = [0] * len(val)
    while i < len(val) and val[i] < ref[0]:
        i += 1
    while j < len(ref) and i < len(val):
        while j < len(ref) - 1 and ref[j + 1] < val[i]:
            j += 1
        res[i] = j
        if j < len(ref) - 1 and ref[j + 1] - val[i] < val[i] - ref[j]:
            # val[i] is nearer to ref[j + 1]
            res[i] = j + 1
        i += 1
    return res

def hz2mel(hz):
    return 1127.01028 * m.log(1 + hz / 700)

def mel2hz(mel):
    return (m.exp(mel / 1127.01028) - 1) * 700

# Generate MFCC filterbank
def mfcc_fb(f, min_hz = 80, max_hz = 8000, n_filt = 40):
    # fundamental frequency of adult male voice is 85 ~ 180 Hz
    # fundamental frequency of adult female is 165 ~ 255 Hz
    # frequency range of adult male voice is upto 8 kHz
    min_mel = hz2mel(min_hz)
    max_mel = hz2mel(max_hz)
    centers = list(m.linspace(min_mel, max_mel, n_filt)[0]) # linear in mel
    left = 2 * centers[0] - centers[1] # left bottom of the first triangle
    right = 2 * centers[-1] - centers[-2] # right bottom of the last triangle
    centers = [left] + centers + [right]
    centers = [mel2hz(x) for x in centers] # convert back to Hz
    
    # create the filterbank matrix
    # number of rows: number of filters
    # number of columns: number of frequency bins in spectrogram, i.e. length of f
    centers = find_nearest(centers, list(m.transpose(f)[0]))
    fb = [[0] * len(f) for i in range(n_filt)]
    for i in range(0, n_filt):
        # left slope, "/" shape
        fb[i][centers[i]:(centers[i + 1] + 1)] = list(m.linspace(0.0, 1.0, centers[i + 1] - centers[i] + 1)[0])
        # right slope, "\" shape
        fb[i][centers[i + 1]:(centers[i + 2] + 1)] = list(m.linspace(1.0, 0.0, centers[i + 2] - centers[i + 1] + 1)[0])

        # normalization of each row
        row_sum = sum(fb[i])
        for j in range(len(f)):
            fb[i][j] /= row_sum
    return fb

def mfcc(s_hz, fb, n_dct = 15):
    # compute the mel power spectrum in dB
    s_mel = m.mtimes(matlab.double(fb), s_hz)
    power = m.db(m.power(m.abs(s_mel), 2.0))

    # compute the DCT of each column of the mel spectrum
    dct = m.dct(power)

    # remove the first coefficient and keep n_dct number of coefficients
    mfcc = dct[1:(n_dct + 1)]

    # normalize
    mean = m.repmat(m.mean(mfcc, 1), m.size(mfcc, 1), 1)
    std = m.repmat(m.std(mfcc, 0, 1), m.size(mfcc, 1), 1)
    mfcc = m.rdivide(m.minus(mfcc, mean), std)
    # mfcc = m.rdivide(m.minus(mfcc, m.repmat(m.mean(mfcc, 1), m.size(mfcc,1), 1)), m.std(mfcc, 0, 2))
    return mfcc


# get the energy for time-domian signal 
def energy(x, fs):
    m_energy = m.ceil((len(x) - win_size) / hop_size)
    padding_len = m_energy * hop_size + win_size - len(x)
    # padding_len = m_energy * 256 + 1024 - len(x)
    x = m.padarray(x, matlab.double([padding_len, 0]), 0, 'post')

    # calculate the energy from 0 to N
    window = m.transpose(m.hamming(win_size))
    x = m.power(m.buffer(x, win_size, hop_size), 2.0)
    E = m.rdivide(m.mtimes(window,x), win_size)
    return E
