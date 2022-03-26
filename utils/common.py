from os.path import isdir, isfile
from os import makedirs, remove
from shutil import rmtree


def force_create_folder(path):
    if isdir(path):
        rmtree(path)
    makedirs(path)


def remove_file(path):
    if isfile(path):
        remove(path)
