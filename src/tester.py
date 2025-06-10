
class Tester:
    def __init__(self, dev_set=None, test_set=None):
        self.dev_set = dev_set
        self.test_set = test_set

    def test_dev(self, dev_set=None):
        self.dev_set = dev_set if dev_set is not None else self.dev_set

    def test(self, test_set=None):
        self.test_set = test_set if test_set is not None else self.dev_set






        
        
