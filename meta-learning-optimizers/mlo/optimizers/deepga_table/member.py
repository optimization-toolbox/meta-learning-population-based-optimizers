import numpy as np

from .sliceops import random_slices, match_slices, multi_slice_add_with_multiplier

class Member:
    r'''
        Initialization:
            initial_phenotype: initial list of parameters
            initial_genotype: initial seed and initial sigma

        Internal attributes:
            genotype: list of seeds
            phenotype: list of parameters

        Methods:
            recreate(new_genotype):
                update genotype and phenotype from new_genotype
            mutate(rng_genes):
                create a new gene and update genotype and phenotype
    '''

    def __init__(self, initial_phenotype, genotype, random_table):

        # Define initial phenotype:
        self.initial_phenotype = initial_phenotype
        self.phenotype = self.initial_phenotype.copy()
        self.random_table = random_table

        # Internal attributes:
        self.rng = np.random.RandomState()
        self.size = len(self.initial_phenotype)

        # Define genotype and phenotype:
        self.recreate(genotype)

    def mutate(self, rng_genes, sigma):
        r'''create a new gene and update genotype and phenotype'''

        # Increase genotype:
        seed = rng_genes.randint(0, 2 ** 32 - 1)
        self.genotype.append([seed, sigma])

        # Mutate phenotype:
        self.rng.seed(seed)

        param_slices = self._draw_random_parameter_slices(self.rng)
        table_slices = self._draw_random_table_slices(self.rng)
        param_slices, table_slices = match_slices(param_slices, table_slices)
        multi_slice_add_with_multiplier(self.phenotype, self.random_table, param_slices, table_slices, sigma)


    def recreate(self, new_genotype):
        r'''update genotype and phenotype from new_genotype'''

        # Set genotype:
        self.genotype = new_genotype[:]

        # Set phenotype:
        self.phenotype[:] = self.initial_phenotype[:]
        for seed, sigma in self.genotype:
            self.rng.seed(seed)

            param_slices = self._draw_random_parameter_slices(self.rng)
            table_slices = self._draw_random_table_slices(self.rng)
            param_slices, table_slices = match_slices(param_slices, table_slices)
            multi_slice_add_with_multiplier(self.phenotype, self.random_table, param_slices, table_slices, sigma)


            #self.phenotype += self.rng.randn(self.size) * sigma


    def _draw_random_parameter_slices(self, rng):
        """
        Chooses a constrained slice subset of the parameters (start, stop, step)
        to give roughly num_mutations perturbations (less if overlap if
        step is too large)
        """

        return random_slices(rng, self.size, self.size, 1)

    def _draw_random_table_slices(self, rng):
        """
        Chooses a constrained slice subset of the RN table (start, stop, step)
        to give roughly num_mutations random numbers (less if overlap if
        step is too large)
        """

        return random_slices(rng, 20000000, self.size, 5)