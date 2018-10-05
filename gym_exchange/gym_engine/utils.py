import collections
import six


def iterable(arg):
    return (isinstance(arg, collections.Iterable) and not
            isinstance(arg, six.string_types))