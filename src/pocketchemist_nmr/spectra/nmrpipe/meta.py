"""
Functions and utilities to unpack NMRPipe headers
"""
import struct
from array import array
import typing as t

from .definitions import get_nmrpipe_definitions
from .constants import header_size_bytes, data_size_bytes
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
        The location to place the file after reading the header. If None,
        the original location will be set.

    Returns
    -------
    nmrpipe_meta
        Return a dict (:obj:`.NMRPipeMetaDict`) with metadata entries from
        an NMRPipe spectrum.
    """

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
        offset = field_locations[key] * data_size_bytes

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


def save_nmrpipe_meta(meta: NMRPipeMetaDict,
                      size_bytes: t.Optional[int] = header_size_bytes,
                      data_size_bytes: int = data_size_bytes) -> bytes:
    """Save an NMRPipe meta dict into a string of bytes that can be used
    as a header for an NMRPipe spectrum.

    Parameters
    ----------
    meta
        A dict (:obj:`.NMRPipeMetaDict`) with metadata entries from
        an NMRPipe spectrum.
    size_bytes
        The size of the header in bytes
    data_size_bytes
        The size of elements (floats) in the header
    """
    # Get the header definitions
    field_locations, field_descriptions, text_fields = get_nmrpipe_definitions()
    fields_by_location = {v: k for k, v in field_locations.items()}

    # Create an empty header
    num_elems = int(size_bytes / data_size_bytes)
    header = array('f', (float(0.0) for i in range(num_elems)))

    # Pack header float values
    for location, field_name in fields_by_location.items():
        offset = location * data_size_bytes

        # Construct the text field name from the field name
        text_field_name = "SIZE_" + field_name.replace('FD', '')

        # print(field_name, meta[field_name])
        if text_field_name in text_fields:
            # Handle strings
            text_size = text_fields[text_field_name]  # in 4-byte floats
            struct.pack_into(f'{text_size}s', header, offset,
                             meta[field_name].encode())
        else:
            # Handle floating-point values
            struct.pack_into('f', header, offset, meta[field_name])

    return header.tobytes()
