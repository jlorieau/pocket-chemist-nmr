"""
Functions to parse NMRPipe headers
"""
import typing as t
from pathlib import Path

from ..meta import NMRMetaDescriptionDict

__all__ = ('get_nmrpipe_definitions',)

#: The NMRPipe fdatap.h file to use
fdatap_filepath = Path(__file__).with_name('fdatap.h')  # same directory

# Cached versions of the definitions dicts

#: The location of fields in the binary header. The field names are the keys,
#: and the locations (as multiples of 4 bytes) as values.
field_locations = None

#: The descriptions of fields. The field names are the keys, and the
#: description strings are the values
field_descriptions = None

#: The identity and size of text fields. The field names are the keys, and
#: the field size (in bytes) are the values.
text_fields = None


class NMRPipeMetaDescriptionDict(NMRMetaDescriptionDict):
    """A metadata description dict for NMRPipe spectra"""


def get_nmrpipe_definitions() -> t.Tuple[dict, dict, dict]:
    """Return a dict of offsets: parameter names.

    This function caches the definition dicts. To reload them, delete the
    file at filename.

    Returns
    -------
    field_locations, field_descriptions, text_fields
        The definitions dicts.
    """
    # See if the definitions dicts have already been processed
    global field_locations, field_descriptions, text_fields
    if all(i is not None for i in (field_locations, field_descriptions,
                                   text_fields)):
        return field_locations, field_descriptions, text_fields

    # They haven't been processed. Process them and save the definitions file
    import re

    # Load fdatap.h
    with open(fdatap_filepath, 'r') as f:
        text = f.read()

        # Select the portion of the file that deals with header locations in
        # 4-byte floats
        m_start = re.search(r'^#define FDMAGIC\s', text, re.MULTILINE)
        m_end = re.search(r'^#define FDF4TDSIZE\s+\d+', text, re.MULTILINE)
        offsets_str = text[m_start.span()[0]:m_end.span()[1]]

        # Select the portion of the file that deals with the size of text
        # characters
        m_start = re.search(r'^#define SIZE_NDLABEL', text, re.MULTILINE)
        m_end = re.search(r'^#define SIZE_TITLE\s+\d+', text, re.MULTILINE)
        text_fields_str = text[m_start.span()[0]:m_end.span()[1]]

    # Create dict for translations in 4-byte locations
    offsets_it = re.finditer(r"#define\s+(?P<name>[\w\d]+)\s+"
                             r"(?P<offset>\d+)"
                             r"(\s*/\*(?P<desc>[^*]+)\*/)?",
                             offsets_str)

    # Prepare the field locations dict
    field_locations, field_descriptions = {}, NMRMetaDescriptionDict()
    for match in offsets_it:
        d = match.groupdict()  # get the match's capture group dict
        name, offset, desc = map(d.get, ('name', 'offset', 'desc'))

        # Set the entries in the dicts
        field_locations[name] = int(offset)

        if desc is not None and desc.strip():
            # Add only non-empty entries
            field_descriptions[name] = desc.strip()

    # Prepare to text fields dict
    text_fields_it = re.finditer(r"#define\s+(?P<name>[\w\d]+)\s+"
                                 r"(?P<size>\d+)", text_fields_str)
    text_fields = {m.groupdict()['name']: int(m.groupdict()['size'])
                   for m in text_fields_it}
    return field_locations, field_descriptions, text_fields
