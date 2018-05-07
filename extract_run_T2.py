import os, matlab.engine, extract
m = matlab.engine.start_matlab()
parent_dir = os.path.dirname(os.getcwd())
extract.run(parent_dir, m, [1024, 512, 24414, 80, 300, 8000], 1, 2, 'T2')
