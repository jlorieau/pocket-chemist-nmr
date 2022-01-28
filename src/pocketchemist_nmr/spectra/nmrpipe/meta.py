"""
Functions and utilities to unpack NMRPipe headers
"""
import struct
import typing as t

from .definitions import get_nmrpipe_definitions
from .constants import header_size_bytes
from ..meta import NMRMetaDict

__all__ = ('load_nmrpipe_meta', 'NMRPipeMetaDict')


class NMRPipeMetaDict(NMRMetaDict):
    """A metadata dict containing entries in NMRPipe format"""


def load_nmrpipe_meta(filelike: t.BinaryIO, start: int = 0,
                      end: t.Optional[int] = header_size_bytes) \
        -> NMRPipeMetaDict:
    """Retrieve metadata in NMRPipe format.

    Parameters
    ----------
    filelike
        A file object for reading in binary mode
    start
        The start byte to start reading the NMRPipe header data.
    end
        The end by to end reading the NMRPipe header data. If this is None,
        the the filelike's position will be placed to its original place.

    Returns
    -------
    nmrpipe_meta
        Return a dict (:obj:`.NMRPipeMetaDict`) with metadata entries from
        an NMRPipe spectrum.
    """
    assert 'b' in filelike.mode, "File must be read in binary mode"

    # Get the header definitions
    field_locations, field_descriptions, text_fields = get_nmrpipe_definitions()
    fields_by_location = {v: k for k, v in field_locations.items()}

    # Get the current offset for the buffer and start the buffer, if specified
    cur_pos = filelike.tell()
    if isinstance(start, int):
        filelike.seek(start)

    # Retrieve the buffer in binary format
    buff = filelike.read(end if end is not None else header_size_bytes)

    # Reset the buffer, if specified, or place it back to where it started
    if isinstance(end, int):
        filelike.seek(end)
    else:
        filelike.seek(cur_pos)

    # Parse the buffer float values
    hdr_it = struct.iter_unpack('f', buff)
    pipedict = {fields_by_location[i]: v for i, (v,) in enumerate(hdr_it)
                if i in fields_by_location}

    # Parse the strings
    for label, size in text_fields.items():
        # Find the string location and size
        key = 'FD' + label.replace('SIZE_', '')

        if key not in field_locations:
            continue

        # Get the offset. This is the number of floats (4-bytes) before the
        # text entry, so it needs to be multiplied by 4
        offset = field_locations[key] * 4

        # Try to convert to string
        try:
            # Locate and unpack the string
            string = struct.unpack_from(f'{size}s', buff, offset=offset)[0]

            # Convert the string to unicode and remove empty bytes
            pipedict[key] = string.decode().strip('\x00')
        except (UnicodeDecodeError, struct.error):
            pass

    # Convert to and return a NMRPipeMetaDict
    return NMRPipeMetaDict(pipedict)
