import math
import numpy as np
import torch
# from decorators import cached


class AbstractTransform:

    def transform(self, src):
        pass

    def transform_sequence(self, src_seq):
        return map(lambda x: self.transform(x), src_seq)

    def inverse_transform(self, src):
        pass

    def inverse_transform_sequence(self, src_seq):
        return map(lambda x: self.inverse_transform(x), src_seq)


class WalshHadamardTransform(AbstractTransform):

    def __init__(self, coeff=None):
        self.__coeff = coeff

    # @cached
    def _get_matrix(self, size):
        # Generating Hadamard matrix of size 'size'
        n = int(math.log(size, 2))
        # row = [1 / (np.sqrt(2) ** n) for _ in range(0, size)]
        row = [1 for _ in range(0, size)]
        matrix = [list(row) for i in range(0, size)]
        for i in range(0, n):
            for j in range(0, size):
                for k in range(0, size):
                    if (j / 2 ** i) % 2 == 1 and (k / 2 ** i) % 2 == 1:
                        matrix[j][k] = -matrix[j][k]
                        if self.__coeff is not None:
                            if matrix[j][k] - self.__coeff < 0.000001:
                                # print('Substitution works! ' + str(matrix[j][k]) + ' ' + str(self.__coeff))
                                matrix[j][k] = 0

        # Producing Walsh-Hadamard matrix by ordering frequencies in ascending order
        matrix.sort(key=lambda x: sum(map(lambda a: a[0] * a[1] < 0, zip(x[1:], x[:-1]))))
        return matrix

    def transform(self, src):
        size = src.shape[1]
        # print(self._get_matrix(size))
        h = torch.tensor(self._get_matrix(size), device=src.device, dtype=torch.float32)
        return torch.abs(src @ h)

    def inverse_transform(self, src):
        return self.transform(src)
