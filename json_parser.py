import json
import pprint
import argparse

class JsonParser(object):
    def __init__(self):
        pass

    def dump_json(self, opt, path):
        json_dir = path + "opt.json"
        argparse_dict = vars(opt)
        with open(json_dir, 'w') as outfile:
            json.dump(argparse_dict, outfile)
        print ("configs have benn dumped into %s" % json_dir)

    def read_json_as_argparse(self, path):
        json_dir = path + "opt.json"
        js = open(json_dir).read()
        data = json.loads(js)
        opt = argparse.Namespace()
        for key, val in data.items():
            setattr(opt, key, val) 
        return opt

        


