# Licensed under a 3-clause BSD style license - see LICENSE.rst
from IPython.parallel import Client
client = Client()
dview = client[0]  # direct view object


@dview.remote(block=True)
def remote_fetch(*args, **kwargs):
    import Ska.engarchive.fetch_eng as fetch
    return fetch.Msid(*args, **kwargs)
