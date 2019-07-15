from inspect import getfullargspec
from functools import wraps
import func_utils as utils
import os

def ModelInfo(func):
    # print('Entry point in decorator')
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = getfullargspec(func)
    default_root = './results'

    def _get_model_path(model_name, save_root):
        if model_name == None:
            if save_root == None:
                model_name, model_path = utils.find_newest_model(default_root) # auto-find the newest model
            else:
                model_name, model_path = utils.find_newest_model(save_root)
        else:
            if save_root == None:
                model_path = os.path.join(default_root, model_name)
            else:
                model_path = os.path.join(save_root, model_name)
        
        print(model_name)
        return model_name, model_path

    @wraps(func)
    def wrap(init_self, *args, **kwargs):
        model_name = None
        save_root = None
        if defaults:
            for var, val in zip(reversed(args), reversed(defaults)):
                print(var, val)
                if var == 'model_name':
                    model_name = val
                if var == 'save_root':
                    save_root = val

        model_name, model_path = _get_model_path(model_name, save_root)

        init_self.model_name = model_name
        init_self.model_path = model_path

        func(init_self, *args, **kwargs)

    return wrap
    # print('Exit point in decorator')

class TT:
    @ModelInfo
    def __init__(self, model_name=None, save_root=None, CHK=True):
        self.a = 1
        self.b = 2
        print('in TT')

    def test(self):
        print(self.model_name)
        print(self.model_path)


if __name__ == "__main__":

    f = TT()

    print ('f.a ==',f.a)
    print ('f.b ==',f.b)

    f.test()
    # print ('f.model_name ==',f.model_name)
    # print ('f.model_path ==',f.model_path)
    print(dir(f))