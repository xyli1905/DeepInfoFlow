

class cfun:
    def __init__(self, p):
        self.param = p
    
    def add_one(self):
        self.param += 1


if __name__ == "__main__":
    D = {1: "cfun", 2:cfun}

    L = []
    L.append(eval(D[1])(p=0))
    L.append(D[2](p=5))

    L[0].add_one()
    print(L[0].param, L[1].param)

    L[1].add_one()
    print(L[0].param, L[1].param)

    print(type(eval(D[2])))
    
