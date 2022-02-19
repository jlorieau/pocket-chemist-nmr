"""
Functions for applying or removing artifacts of digital filters from data
collected on Bruker spectrometers
"""
from math import floor
import typing as t

import torch

#: Data copied from
#: https://github.com/jjhelmus/nmrglue/blob/master/nmrglue/fileio/bruker.py
#: This table gives the number of gap points (integer value) and first-order
#: phase correct (decical part) for DSP Bruker time-domain data
#: The values fo this table originally come from
#: - W. M. Westler and F.  Abildgaard. DMX DIGITAL FILTERS AND NON-BRUKER
#:   OFFLINE PROCESSING III. 1996
#:   https://web.archive.org/web/20040130234756/
#:   http://garbanzo.scripps.edu:80/nmrgrp/wisdom/dig.nmrfam.txt
#: - https://web.archive.org/web/20210304082847/
#:   http://sbtools.uchc.edu/help/nmr/nmr_toolkit/bruker_dsp_table.asp
#: The table is organized as follors:
#: {'DSPFVS' : {'DECIM': value}}
bruker_dsp_table = {
    10: {
        2: 44.75,
        3: 33.5,
        4: 66.625,
        6: 59.083333333333333,
        8: 68.5625,
        12: 60.375,
        16: 69.53125,
        24: 61.020833333333333,
        32: 70.015625,
        48: 61.34375,
        64: 70.2578125,
        96: 61.505208333333333,
        128: 70.37890625,
        192: 61.5859375,
        256: 70.439453125,
        384: 61.626302083333333,
        512: 70.4697265625,
        768: 61.646484375,
        1024: 70.48486328125,
        1536: 61.656575520833333,
        2048: 70.492431640625,
    },
    11: {
        2: 46.,
        3: 36.5,
        4: 48.,
        6: 50.166666666666667,
        8: 53.25,
        12: 69.5,
        16: 72.25,
        24: 70.166666666666667,
        32: 72.75,
        48: 70.5,
        64: 73.,
        96: 70.666666666666667,
        128: 72.5,
        192: 71.333333333333333,
        256: 72.25,
        384: 71.666666666666667,
        512: 72.125,
        768: 71.833333333333333,
        1024: 72.0625,
        1536: 71.916666666666667,
        2048: 72.03125
    },
    12: {
        2: 46.,
        3: 36.5,
        4: 48.,
        6: 50.166666666666667,
        8: 53.25,
        12: 69.5,
        16: 71.625,
        24: 70.166666666666667,
        32: 72.125,
        48: 70.5,
        64: 72.375,
        96: 70.666666666666667,
        128: 72.5,
        192: 71.333333333333333,
        256: 72.25,
        384: 71.666666666666667,
        512: 72.125,
        768: 71.833333333333333,
        1024: 72.0625,
        1536: 71.916666666666667,
        2048: 72.03125
    },
    13: {
        2: 2.75,
        3: 2.8333333333333333,
        4: 2.875,
        6: 2.9166666666666667,
        8: 2.9375,
        12: 2.9583333333333333,
        16: 2.96875,
        24: 2.9791666666666667,
        32: 2.984375,
        48: 2.9895833333333333,
        64: 2.9921875,
        96: 2.9947916666666667
    }
}


def bruker_group_delay(grpdly: t.Optional[float] = None,
                       dspfvs: t.Optional[int] = None,
                       decim: t.Optional[float] = None) -> \
        t.Union[None, float]:
    """Retrieve the group delay based on the digital acquisition parameters.

    Parameters
    ----------
    grpdly
        The Bruker Group Delay paramter. Either this parameter or dspfvs and
        decim must be specified.
    dspfvs
        Bruker firmware version. Either this parameter and decim must be
        specified or grpdly must be specified.
    decim
        Bruker decimation rate. Either this parameter and dspfvs must be
        specified or grpdly must be specified

    Returns
    -------
    group_delay
        The group delay. (The number of points to left-circular shift and
        the first-order phase to apply.)
    """
    assert grpdly is not None or (dspfvs is not None and decim is not None), (
        "Either 'grpdly' or 'dspfvs' and 'decim' must be specified.")
    # Determine the left circular shift

    # Get the group delay from the lookup table, if it wasn't specified
    if grpdly is None:
        try:
            grpdly = bruker_dsp_table[dspfvs][decim]
        except KeyError:
            raise KeyError(f"A group delay for a DSPFVS of '{dspfvs}' and "
                           f"DECIM of '{decim}' could not be found.")
    return grpdly

