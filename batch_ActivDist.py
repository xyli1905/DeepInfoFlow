from ActivationDist import ActivationDist
import os



if __name__ == "__main__":

    save_root = '/Users/xyli1905/Desktop/exp_ADAM'

    for d in os.listdir(save_root):
        bd = os.path.join(save_root, d)
        if os.path.isdir(bd):
            act = ActivationDist(model_name = d, save_root = save_root)
            act.CalculateDist()
