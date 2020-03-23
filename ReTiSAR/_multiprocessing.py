# Explicit multiprocessing context using the fork start method. This exists as a compat layer now
# that Python 3.8 has changed the default start method for macOS to ``spawn`` which is
# incompatible with our code base currently.
#
# according to https://github.com/ansible/ansible/pull/63581

__metaclass__ = type

import multiprocessing

try:
    mp_context = multiprocessing.get_context("fork")

    # # TODO: "Forkserver" would be the recommended functionality according to
    # #  https://docs.python.org/3/library/multiprocessing.html
    # #
    # # Anyhow, these options do not (currently) work, due to the underlying data serialization
    # # happening when instantiating a new process. Data pickling does not (naturally) work for
    # # non-standard data types / classes. In our case, `jack.Client` and `jack.Port` with the
    # # underlying C data types and pointers are the problem.
    # #
    # # It is unclear whether pickling could be implemented for this kind of external data, see
    # # https://docs.python.org/3/library/pickle.html#persistence-of-external-objects and
    # # https://stackoverflow.com/questions/36301322/pickle-cython-class-with-c-pointers
    # #
    # # Something like this could be a possible implementation in `jack.Client`?
    # #
    # # def __getstate__(self):
    # #     # Copy the object's state from self.__dict__ which contains
    # #     # all our instance attributes. Always use the dict.copy()
    # #     # method to avoid modifying the original state.
    # #     state = self.__dict__.copy()
    # #
    # #     # Remove the unpicklable entries.
    # #     del(state['_ptr'])
    # #     del(state['_position'])
    # #
    # #     # import pickle
    # #     # for key in list(state.keys()):
    # #     #     try:
    # #     #         pickle.dumps(state[key].copy())
    # #     #     except (pickle.PicklingError, AttributeError, TypeError):
    # #     #         print(f'key "{key}" could not be pickled.')
    # #     #         del(state[key])
    # #     #     else:
    # #     #         print(f'key "{key}" good.')
    # #
    # #     return state
    # #
    # # def __setstate__(self, state):
    # #     # Restore the pickled instance attributes.
    # #     self.__dict__.update(state)
    # #
    # #     # Restore the unpicklable instance attributes.
    # #     self._ptr = ...
    # #     self._position = ...
    # #     # How would that be possible?
    # #
    # mp_context = multiprocessing.get_context("spawn")
    # mp_context = multiprocessing.get_context("forkserver")
except AttributeError:
    mp_context = multiprocessing
