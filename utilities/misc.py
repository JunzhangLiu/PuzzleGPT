import errno
import os


def save_config(cfg, path):
    with open(path, 'w') as f:
        f.write(cfg.dump())

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise