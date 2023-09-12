

class A:

    def __init__(self, x) -> None:
        self.x = x

def x():
    a = A("adad")
    b = A(a)
    del b.x
    return b

b = x()
print(b.x.x)
