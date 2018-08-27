from contextlib import contextmanager
import time
import yaml
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import KFold, StratifiedKFold


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


class WorkSpaceError(Exception):
    pass


class WorkSpace:
    def __init__(self, name, path='./workdir'):
        self.__name = name
        self.__path = os.path.abspath(path)
        self.__dir = os.path.join(self.__path, self.__name)
        if not os.path.exists(self.__dir):
            os.mkdir(self.__dir)
        else:
            pass

    def load(self, filename):
        i = filename.rfind('.') + 1
        extension = filename[i:]
        filename = os.path.join(self.__dir, filename)
        if extension == 'pkl':
            with open(filename, 'rb') as f:
                ret = pickle.load(f)
        elif extension == '.csv':
            ret = pd.read_csv(filename)
        else:
            raise WorkSpaceError(
                'load file with unsupport type %s .' % filename)

        return ret

    def load_model(self):
        filenames = ['kfold_model.pkl', 'single_model.pkl', 'stacking.pkl']
        ret = None
        for filename in filenames:
            fullname = os.path.join(self.__dir, filename)
            if os.path.exists(fullname):
                ret = self.load(filename)
                break
        return ret

    def save(self, obj, filename):
        i = filename.rfind('.') + 1
        extension = filename[i:]
        filename = os.path.join(self.__dir, filename)
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
        elif type_ == 'stacking':
            src = './notebook/TemplateReport/Stacking.ipynb'
        dst = os.path.join(self.__dir, 'report.ipynb')
        shutil.copyfile(src, dst)


def read_config(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data


def gen_cv(num_folds, random_state, stratified):
    if stratified:
        return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=num_folds, shuffle=True, random_state=random_state)


if __name__ == '__main__':
    pass
