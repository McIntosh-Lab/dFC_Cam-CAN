from os import makedirs, path

def make_dir_if_missing(path_str):
    if not path.exists(path_str):
        makedirs(path_str)

class Dict2Object:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
