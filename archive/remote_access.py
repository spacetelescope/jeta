# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Settings and functions for remotely accessing an engineering archive.
"""
from __future__ import print_function, division, absolute_import

import sys
import os
import getpass
try:
    import ipyparallel as parallel
except ImportError:
    from IPython import parallel
from six.moves import input

# To use remote access, this flag should be set True (it is true by default
# on Windows systems, but can manually be set to true on Linux systems
# if you don't have direct access to the archive)
access_remotely = sys.platform.startswith('win')

# Hostname (IP), username, and password for remote access to the eng archive
hostname = None
username = None
password = None

# Flag to ask the user for credentials if the ssh login fails
ask_again_if_connect_fails = True

# Flag to show print output for remote calls
show_print_output = sys.platform.startswith('win')

# Client key file for connecting to the remote server (ipcontroller)
client_key_file = os.path.join(sys.prefix,"ska_remote_access.json")

# IPython parallel client for accessing the remote python engine
_remote_client = None

class RemoteConnectionError(Exception):
    pass

def establish_connection():
    """
Function to establish a connection to the remote server
Return success status (True/False)
"""
    global _remote_client
    global hostname
    global username
    global password
    
    try_to_connect = True
    
    # Loop until the user is able to connect or cancels
    while _remote_client is None:
        # Get the username and password if not already set
        hostname = hostname or input('Enter hostname (or IP) of Ska ' +
                                         'server (enter to cancel):')
        if hostname=="":
            break
        default_username = getpass.getuser()
        username = username or input('Enter your login username [' + 
                                         default_username + ']:')
        password = password or getpass.getpass('Enter your password:')
        
        # Open the connection to the server
        print('Establishing connection to ' + hostname + '...')
        sys.stdout.flush()
        try:
            _remote_client = parallel.Client(client_key_file,
                                             sshserver=username+'@'+hostname,
                                             password=password)
        except:
            print('Error connecting to server ',hostname,': ',sys.exc_info()[0])
            sys.stdout.flush()
            _remote_client = None
            # Clear out information so the user can try again
            hostname = None
            username = None
            password = None
            if not ask_again_if_connect_fails:
                break
        
    return(connection_is_established())


def connection_is_established():
    """
Function to check if a connection to the remote server has been established
"""
    return (not _remote_client is None)
    

def execute_remotely(fcn, *args, **kwargs):
    """
Function for executing a function remotely
"""
    if not connection_is_established():
        raise parallel.ConnectionError(
                "Connection not established to remote server")
    dview = _remote_client[0]; # Use the first (and should be only) engine
    dview.block = True
    return dview.apply_sync(fcn, *args, **kwargs)
    

def test_connection():
    """
Function to perform a quick test of the connection to the
remote server
"""
    def remote_fcn():
        import os
        return os.sys.version
    return (execute_remotely(remote_fcn))


def close_connection():
    """
Function to close the connection to the remote server
"""
    global _remote_client
    
    _remote_client.close()
    _remote_client = None
    
