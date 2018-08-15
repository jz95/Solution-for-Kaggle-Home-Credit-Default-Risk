from contextlib import contextmanager
import time
import yaml
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import shutil


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


class WorkSpaceError(Exception):
    pass


class WorkSpace:
    def __init__(self, name, path='./workdir'):
        self._name = name
        self._path = os.path.abspath(path)
        self._dir = os.path.join(self._path, self._name)
        self._logger = None
        if not os.path.exists(self._dir):
            os.mkdir(self._dir)
        else:
            pass

    def load(self, filename):
        i = filename.rfind('.') + 1
        extension = filename[i:]
        filename = os.path.join(self._dir, filename)
        if extension == 'pkl':
            with open(filename, 'rb') as f:
                ret = pickle.load(f)
        elif extension == '.csv':
            ret = pd.read_csv(filename)
        else:
            raise WorkSpaceError(
                'load file with unsupport type %s .' % filename)

        return ret

    def save(self, obj, filename):
        i = filename.rfind('.') + 1
        extension = filename[i:]
        filename = os.path.join(self._dir, filename)
        if extension == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
        elif extension == 'csv' and isinstance(obj, pd.DataFrame):
            obj.to_csv(filename, index=False)
        elif extension in ['png', 'jpg', 'jpeg']:
            plt.savefig(filename)
        else:
            with open(filename, 'w') as f:
                f.write(obj)

    def gen_report(self, type_):
        if type_ == 'kfold':
            src = './notebook/TemplateReport/Kfold.ipynb'
        elif type_ == 'grid_search':
            src = './notebook/TemplateReport/GridSearch.ipynb'
        dst = os.path.join(self._dir, 'report.ipynb')
        shutil.copyfile(src, dst)


def read_config(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data


if __name__ == '__main__':
    pass
