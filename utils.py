import os
import json
import argparse


def save_opt_to_json(opt, mpath):
    json_dir = os.path.join(mpath, "opt.json")
    argparse_dict = vars(opt)
    with open(json_dir, 'w') as outfile:
        json.dump(argparse_dict, outfile)
    print ("configs have been dumped into %s" % json_dir)

def load_json_as_argparse(mpath):
    try:
        json_dir = os.path.join(mpath, "opt.json")
        js = open(json_dir).read()
        data = json.loads(js)
        opt = argparse.Namespace()
        for key, val in data.items():
            setattr(opt, key, val) 
        return opt
    except Exception as e:
        print("No such file or directory %s" % (json_dir))

# auto find the most recent model, used in ComputeMI and plot_utils
def find_newest_model(mpath):
    all_subdirs = []
    for d in os.listdir(mpath):
        bd = os.path.join(mpath, d)
        if os.path.isdir(bd): all_subdirs.append(bd)
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    mname = os.path.split(latest_subdir)[-1]
    return mname, latest_subdir



if __name__ == "__main__":
    pass
