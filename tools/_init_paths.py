import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_paths = [osp.join(this_dir, '..', 'src'), osp.join(this_dir, '..', 'src', 'lib')]

for lib_path in lib_paths:
    add_path(lib_path)