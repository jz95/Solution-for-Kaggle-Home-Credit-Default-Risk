from contextlib import contextmanager
import time
import yaml
import os
import pickle


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def make_workspace(name):
    filename = './workdir/' + name
    while os.path.exists(filename):
        filename += '_DUP'
    os.mkdir(filename)
    return filename


def get_workspace(name):
    filename = './workdir/' + name
    if not os.path.exists(filename):
        raise NameError('Not found model %s.' % name)
    return filename


def read_config(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data


def save_model(models, workspace):
    with open(workspace + '/models.pkl', 'wb') as f:
        pickle.dump(models, f)


def load_model(workspace):
    with open(workspace + '/models.pkl', 'rb') as f:
        models = pickle.load(f)
    return models


def summary():
    pass


if __name__ == '__main__':
    # read_config('./config/model_config.json')
    pass
