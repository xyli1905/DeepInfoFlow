import json
import pprint
import argparse
import os

class JsonParser(object):
    def __init__(self):
        pass

    def dump_json(self, opt, path):
        json_dir = os.path.join(path, "opt.json")
        argparse_dict = vars(opt)
        with open(json_dir, 'w') as outfile:
            json.dump(argparse_dict, outfile)
        print ("configs have been dumped into %s" % json_dir)

    def read_json_as_argparse(self, path):
        try:
            json_dir = os.path.join(path, "opt.json")
            js = open(json_dir).read()
            data = json.loads(js)
            opt = argparse.Namespace()
            for key, val in data.items():
                setattr(opt, key, val) 
            return opt
        except Exception as e:
            print("No such file or directory %s" % (json_dir))
        

        


