"""
Functions and utilities to unpack NMRPipe headers
"""
import struct

from .definitions import get_nmrpipe_locations

header_size = 2048  # bytes


def load_nmrpipe_meta(filelike, start=0, end=header_size):
    """Retrieve metadata in NMRPipe format."""
    assert 'b' in filelike.mode, "File must be read in binary mode"

    # Get the header definitions
    field_locations, text_fields = get_nmrpipe_locations()
    fields_by_location = {v: k for k, v in field_locations.items()}

    # Get the current offset for the buffer and start the buffer, if specified
    cur_pos = filelike.tell()
    if isinstance(start, int):
        filelike.seek(start)

    # Retrieve the buffer in binary format
    buff = filelike.read(end if end is not None else header_size)

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

    return pipedict
