import os, matlab.engine, extract
m = matlab.engine.start_matlab()

# fundamental frequency of adult male voice is 85 ~ 180 Hz
# fundamental frequency of adult female is 165 ~ 255 Hz
# frequency range of adult male voice is upto 8 kHz

parent_dir = os.path.dirname(os.getcwd())
#extract.run(parent_dir, m, [1024, 512, 24414, 80, 300, 8000], 1, 2)
#extract.run(parent_dir, m, [1024, 512, 24414, 80, 300, 8000], 1, 2, False)

extract.run(parent_dir, m, [1024, 512, 24414, 80, 300, 8000], 7, 50)
extract.run(parent_dir, m, [1024, 512, 24414, 80, 300, 8000], 7, 9, False)
