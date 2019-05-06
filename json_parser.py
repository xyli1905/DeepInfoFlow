import json
import pprint
import argparse

class JsonParser(object):
    def __init__(self, opt, dir):
        self.opt = opt
        self.dir = dir

    def dump_json(self):
        json_dir = self.dir + "opt.json"
        argparse_dict = vars(self.opt)
        with open(json_dir, 'w') as outfile:
            json.dump(argparse_dict, outfile)
        print ("configs have benn dumped into %s" % json_dir)

    def read_json_as_argparse(self):
        json_dir = self.dir + "opt.json"
        js = open(json_dir).read()
        data = json.loads(js)
        opt = argparse.Namespace()
        for key, val in data.items():
            setattr(opt, key, val) 
        return opt

        


