# Explicit multiprocessing context using the fork start method. This exists as a compat layer now
# that Python 3.8 has changed the default start method for macOS to ``spawn`` which is
# incompatible with our code base currently.
#
# according to https://github.com/ansible/ansible/pull/63581

__metaclass__ = type

import multiprocessing

try:
    mp_context = multiprocessing.get_context("fork")
    # mp_context = multiprocessing.get_context("spawn")
except AttributeError:
    mp_context = multiprocessing
