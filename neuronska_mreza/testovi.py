import unittest
import util
import seaborn as sns


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_neuron(self):
        df = sns.load_dataset('iris')

        return ''


if __name__ == '__main__':
    unittest.main()
