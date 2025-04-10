import typing
import jax.random as jr
import pytest


class KeyStream:
    def __init__(self, seed: int = 0):
        self.counter = 0
        self.key = jr.key(seed)

    def __call__(self):
        self.counter += 1
        return jr.fold_in(self.key, self.counter)


@pytest.fixture
def nextkey():
    return KeyStream(seed=0)
