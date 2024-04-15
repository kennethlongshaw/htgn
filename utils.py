from operator import itemgetter
from itertools import islice


def iter_index(iterable, value, start=0, stop=None):
    "Return indices where a value occurs in a sequence or iterable."
    # iter_index('AABCADEAF', 'A') → 0 1 4 7
    seq_index = getattr(iterable, 'index', None)
    if seq_index is None:
        # Path for general iterables
        it = islice(iterable, start, stop)
        for i, element in enumerate(it, start):
            if element is value or element == value:
                yield i
    else:
        # Path for sequences with an index() method
        stop = len(iterable) if stop is None else stop
        i = start
        try:
            while True:
                yield (i := seq_index(value, i, stop))
                i += 1
        except ValueError:
            pass


def filter_by_index(data: list, index: list, selection: int):
    mask = list(iter_index(iterable=index, value=selection))
    return [data[i] for i in mask]
