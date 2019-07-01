class base:
    def __init__(self):
        pass
    
    def F(self):
        print('base')


class child(base):
    def __init__(self):
        super(child, self).__init__()
    
    def F(self, input):
        print('child'+input)



if __name__ == "__main__":
    b = base()
    c = child()

    c.F('test')
    print("test")
    b.F()