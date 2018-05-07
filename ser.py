import matlab

def get_matlab_engine(matlab_engine, arguments):
    global m
    global win_size
    global hop_size
    global window
    global target_fs
    global min_hz
    global max_fund_hz
    global max_hz
    global min_mel
    global max_mel
    global py_min_lag
    global py_max_lag
    
    m = matlab_engine
    [win_size, hop_size, target_fs, min_hz, max_fund_hz, max_hz] = [float(x) for x in arguments]
    window = m.transpose(m.hamming(win_size))
    min_mel = hz2mel(min_hz)
    max_mel = hz2mel(max_hz)
    py_min_lag = float(int(target_fs / max_fund_hz))
    py_max_lag = float(int(target_fs / min_hz))

def mat2list(matlab_double):
    if type(matlab_double) == float:
        return [matlab_double]
    if len(matlab_double) == 1:
        return list(matlab_double[0])
    return [list(x) for x in matlab_double]

def audioread(filename):
    x, fs = m.audioread(filename, nargout = 2) # all audio files are mono
    if fs != target_fs:
        x = m.resample(x, target_fs, fs)
        fs = target_fs
    return x, fs

def spectrogram(x, fs):
    return m.spectrogram(x, m.hamming(win_size), win_size - hop_size, matlab.double(), fs, nargout = 2)

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
def mfcc_fb(f, n_filt = 40):
    centers = mat2list(m.linspace(min_mel, max_mel, n_filt)) # linear in mel
    left = 2 * centers[0] - centers[1] # left bottom of the first triangle
    right = 2 * centers[-1] - centers[-2] # right bottom of the last triangle
    centers = [left] + centers + [right]
    centers = [mel2hz(x) for x in centers] # convert back to Hz
    
    # create the filterbank matrix
    # number of rows: number of filters
    # number of columns: number of frequency bins in spectrogram, i.e. length of f
    centers = find_nearest(centers, mat2list(m.transpose(f)))
    fb = [[0] * len(f) for i in range(n_filt)]
    for i in range(n_filt):
        # left slope, "/" shape
        fb[i][centers[i]:(centers[i + 1] + 1)] = mat2list(m.linspace(0.0, 1.0, centers[i + 1] - centers[i] + 1))
        # right slope, "\" shape
        fb[i][centers[i + 1]:(centers[i + 2] + 1)] = mat2list(m.linspace(1.0, 0.0, centers[i + 2] - centers[i + 1] + 1))

        # normalization of each row
        row_sum = sum(fb[i])
        for j in range(len(f)):
            if row_sum != 0:
                fb[i][j] /= row_sum
    return matlab.double(fb)

def mfcc(s_hz, fb, n_dct = 15):
    # compute the mel power spectrum in dB
    s_mel = m.mtimes(fb, s_hz)
    power = m.db(m.power(m.abs(s_mel), 2.0))

    # compute the DCT of each column of the mel spectrum
    dct = m.dct(power)

    # remove the first coefficient and keep n_dct number of coefficients
    mfcc = dct[1:(n_dct + 1)]

    # normalize
    mean = m.repmat(m.mean(mfcc, 1), len(mfcc), 1)
    std = m.repmat(m.std(mfcc, 0, 1), len(mfcc), 1)
    mfcc = m.rdivide(m.minus(mfcc, mean), std)
    return mfcc

# get the energy for time-domain signal
def energy(x, win_count):
    target_len = (win_count - 1) * hop_size + win_size
    if target_len > len(x):
        X = m.vertcat(x, m.zeros(target_len, len(x), 1))
    else:
        X = x
    X = [v[0] for v in X]
    E = [0] * win_count
    for i in range(win_count):
        E[i] = sum([ \
            X[i * int(hop_size) + j] * X[i * int(hop_size) + j] * window[0][j] \
            for j in range(int(win_size))]) / win_size
    return matlab.double(E) # discard the last column since it is all zero

# with buffer. 6 times slower
def energy_buf(x):
    buffer = m.buffer(x, win_size, win_size - hop_size, 'nodelay')
    x = m.times(buffer, buffer)
    E = m.rdivide(m.mtimes(window, x), win_size)
    return matlab.double(E[0][:-1]) # discard the last column since it is all zero

## pitch version
# the input parameter s is the spectrogram and fs is the time-domain frequency
def pitch(s, fs):
    min_lag = py_min_lag + 1.0
    max_lag = py_max_lag + 0.0
    # window = m.hamming(win_size)
    S = m.real(m.ifft(m.power(m.abs(s), 2.0), win_size, 1.0))
    S = S[int(py_min_lag):int(py_max_lag)]     # slice in the row
    divide_factor = m.transpose(m.minus(win_size, m.linspace(min_lag, max_lag, (max_lag - min_lag + 1))))
    rx = m.rdivide(S, m.repmat(divide_factor, 1.0, len(S[0])))

    # find pitch for each time window
    foo, index = m.max(rx, matlab.double(), 1, nargout = 2)
    return matlab.double([fs / (i + py_min_lag) for i in index[0]])
