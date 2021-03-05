import numpy as np

class RandomGenerator:

    def __init__(self, types, idxs, seed):

        self.types = types
        self.min_idx, self.max_idx = idxs

        self.np_gen = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        curr_type = self.np_gen.choice(self.types, 1)[0]
        curr_idx = self.np_gen.randint(self.min_idx, self.max_idx + 1)
        return curr_type, curr_idx


class DeterministicGenerator:

    def __init__(self, types, idxs):

        self.types = types
        self.min_idx, self.max_idx = idxs

        self.generator = self.chain()

    def chain(self):
        while True:
            for i in range(self.min_idx, self.max_idx + 1):
                for t in self.types:
                    yield t, i

    def __next__(self):
        return next(self.generator)

    def __iter__(self):
        return self


#if __name__ == "__main__":

#    random_gen = RandomGenerator(['A', 'B', 'C'], [2, 4])
#    deterministic_gen = DeterministicGenerator(['A', 'B', 'C'], [2, 4])

#    print()
#    print('==================================================')
#    print('Testing Generators...')
#    print('==================================================')
#    print("Configuration: (['A', 'B'], [2, 4])")
#    print("--------------------------------------------------")
#    print('Deterministic Generator | Random Generator')
#    print("--------------------------------------------------")
#    nb = 15
#    for s1, s2 in zip(deterministic_gen, random_gen):
#        print(s1, " | ", s2)
#        nb -= 1
#        if nb == 0:
#            break
#    print('==================================================')
#    print()
