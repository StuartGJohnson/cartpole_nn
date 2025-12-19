import unittest
import numpy as np

def tuple_fun():
    return 7, 5

class MyTestCase(unittest.TestCase):
    def test_something(self):
        a, *_ = tuple_fun()
        print(a)

    def test_reshape(self):
        a = np.random.rand(2,3,2)
        print("a", a)
        b = np.reshape(a,(6,2))
        print("b", b)
        c = np.reshape(b, (2,3,2))
        print("c", c)



if __name__ == '__main__':
    unittest.main()
