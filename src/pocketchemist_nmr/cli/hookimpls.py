"""
CLI command implementations
"""
from pocketchemist.hookimpls import pocketchemist

from .nmrpipe import nmrpipe
from .gui import nmrdesk


@pocketchemist
def add_command(root_command):
    """Add the CLI subcommands"""
    root_command.add_command(nmrdesk)  # NMRDesk
    root_command.add_command(nmrpipe)  # NMRPipe
