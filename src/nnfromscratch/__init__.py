
from .__nnfromscratch import initialize_network

__all__ = ['initialize_network']


# # In case we need to document a parent class, try this
# from .mess import MyClass, MyOtherClass
#
# __all_exports = [MyClass, MyOtherClass]
#
# for e in __all_exports:
#     e.__module__ = __name__
#
# __all__ = [e.__name__ for e in __all_exports]
