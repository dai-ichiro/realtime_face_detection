import tarfile

data_file = 'VOCtrainval_11-May-2012.tar'
with tarfile.open(data_file) as tar:
    tar.extractall(path='.')