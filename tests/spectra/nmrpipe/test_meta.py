"""
Tests for the NMRPipe metadata functions
"""
from pathlib import Path
import io

import pytest
from pocketchemist_nmr.spectra.meta import NMRMetaDict
from pocketchemist_nmr.spectra.nmrpipe.meta import (load_nmrpipe_meta,
                                                    save_nmrpipe_meta)


spectra_exs = {Path('data') / 'bruker' /
               'CD20170124_av500hd_101_ubq_hsqcsi2d' /
               'hsqcetfpf3gpsi2.ft2': {
    # Values from show_hdr
    'FDMAGIC': 0.0,
    'FDFLTFORMAT': 4008636160.,
    'FDDIMCOUNT': 2.0,
    'FDF2LABEL': 'HN',
    'FDF1LABEL': '15N',
    'FDDIMORDER1': 1.0,
    'FDDIMORDER2': 2.0,
    'FDDIMORDER3': 3.0,
    'FDDIMORDER4': 4.0,
    'FDDMXVAL': 67.98423767089844,
    'FDDMXFLAG': 1.0,
    'FDDELTATR': 0.0,
    'FDNUSDIM': 0.0,
    'FDF3APOD': 0.0,
    'FDF3QUADFLAG': 1.0,
    'FDF4APOD': 0.0,
    'FDF4QUADFLAG': 1.0,
    'FDF1QUADFLAG': 1.0,
    'FDF2QUADFLAG': 1.0,
    'FDPIPEFLAG': 0.0,
    'FDF3UNITS': 0.0,
    'FDF4UNITS': 0.0,
    'FDF3P0': 0.0,
    'FDF3P1': 0.0,
    'FDF4P0': 0.0,
    'FDF4P1': 0.0,
    'FDF2AQSIGN': 0.0,
    'FDPARTITION': 0.0,
    'FDF2CAR': 4.7729997634887695,
    'FDF1CAR': 117.56600189208984,
    'FDF3CAR': 0.0,
    'FDF4CAR': 0.0,
    'FDUSER1': 0.0, 'FDUSER2': 0.0, 'FDUSER3': 0.0, 'FDUSER4': 0.0,
    'FDUSER5': 0.0,
    'FDPIPECOUNT': 0.0,
    'FDUSER6': 0.0,
    'FDFIRSTPLANE': 0.0,
    'FDLASTPLANE': 0.0,
    'FDF2CENTER': 1401.0,
    'FDF1CENTER': 185.0,
    'FDF3CENTER': 1.0,
    'FDF4CENTER': 1.0,
    'FDF2APOD': 160.0,
    'FDF2FTSIZE': 4096.0,
    'FDREALSIZE': 640.0,
    'FDF1FTSIZE': 368.0,
    'FDSIZE': 368.0,
    'FDF2SW': 2003.2052001953125,
    'FDF2ORIG': 3123.397216796875,
    'FDQUADFLAG': 1.0,
    'FDF2ZF': -4096.0,
    'FDF2P0': 145.0,
    'FDF2P1': 0.0,
    'FDF2LB': 0.0,
    'FDF2OBS':  499.87200927734375,
    'FDMCFLAG': 0.0,
    'FDF2UNITS': 0.0,
    'FDNOISE': 0.0,
    'FDTEMPERATURE': 0.0,
    'FDPRESSURE': 0.0,
    'FDRANK': 0.0,
    'FDTAU': 0.0,
    'FDF3FTSIZE': 0.0,
    'FDF4FTSIZE': 0.0,
    'FDF1OBS': 50.65700149536133,
    'FDSPECNUM': 1024.0,
    'FDF2FTFLAG': 1.0,
    'FDTRANSPOSED': 1.0,
    'FDF1FTFLAG': 1.0,
    'FDF1SW': 1671.6820068359375,
    'FDF1UNITS': 0.0,
    'FDF1LB': 0.0,
    'FDF1P0': -90.0,
    'FDF1P1': 0.0,
    'FDMAX': 822781.125000,
    'FDMIN': -437039.906250,
    'FDF1ORIG': 5124.24267578125,
    'FDSCALEFLAG': 0.0,
    'FDDISPMAX': 822781.125000,
    'FDDISPMIN': -437039.906250,
    'FDPTHRESH': 0.0,
    'FDNTHRESH': 0.0,
    'FD2DPHASE': 2.0,
    'FDF2X1': 649.0,
    'FDF2XN': 1672.0,
    'FDF1X1': 0.0,
    'FDF1XN': 0.0,
    'FDF3X1': 0.0,
    'FDF3XN': 0.0,
    'FDF4X1': 0.0,
    'FDF4XN': 0.0,
    'FDDOMINFO': 0.0,
    'FDMETHINFO': 0.0,
    'FDHOURS': 0.0,
    'FDMINS': 0.0,
    'FDSECS': 0.0,
    'FDSRCNAME': '',
    'FDUSERNAME': '',
    'FDMONTH': 0.0,
    'FDDAY': 0.0,
    'FDYEAR': 0.0,
    'FDTITLE': '',
    'FDCOMMENT': '',
    'FDLASTBLOCK': 0.0,
    'FDCONTBLOCK': 0.0,
    'FDBASEBLOCK': 0.0,
    'FDPEAKBLOCK': 0.0,
    'FDBMAPBLOCK': 0.0,
    'FDHISTBLOCK': 0.0,
    'FD1DBLOCK': 0.0,
    'FDSCORE': 0.0,
    'FDSCANS': 0.0,
    'FDF3LB': 0.0,
    'FDF4LB': 0.0,
    'FDF2GB': 0.0,
    'FDF1GB': 0.0,
    'FDF3GB': 0.0,
    'FDF4GB': 0.0,
    'FDF2OBSMID': 0.0,
    'FDF1OBSMID': 0.0,
    'FDF3OBSMID': 0.0,
    'FDF4OBSMID': 0.0,
    'FDF2GOFF': 0.0,
    'FDF1GOFF': 0.0,
    'FDF3GOFF': 0.0,
    'FDF4GOFF': 0.0,
    'FDF2TDSIZE': 640.0,
    'FDF1TDSIZE': 184.0,
    'FDF3TDSIZE': 0.0,
    'FDF4TDSIZE': 0.0,
    'FD2DVIRGIN': 1.0,
    'FDF3APODCODE': 0.0,
    'FDF3APODQ1': 0.0,
    'FDF3APODQ2': 0.0,
    'FDF3APODQ3': 0.0,
    'FDF3C1': 0.0,
    'FDF4APODCODE': 0.0,
    'FDF4APODQ1': 0.0,
    'FDF4APODQ2': 0.0,
    'FDF4APODQ3': 0.0,
    'FDF4C1': 0.0,
    'FDF2APODCODE': 1.0,
    'FDF1APODCODE': 1.0,
    'FDF2APODQ1': 0.44999998807907104,
    'FDF2APODQ2': 0.949999988079071,
    'FDF2APODQ3': 1.0,
    'FDF2C1': -0.5,
    'FDF2APODDF': 0.0,
    'FDF1APODQ1': 0.44999998807907104,
    'FDF1APODQ2': 0.949999988079071,
    'FDF1APODQ3': 1.0,
    'FDF1C1': -0.5,
    'FDF1APOD': 184.0,
    'FDF1ZF': -368.0,
    'FDF3ZF': 0.0,
    'FDF4ZF': 0.0,
    'FDFILECOUNT': 1.0,
    'FDSLICECOUNT0': 1024.0,
    'FDTHREADCOUNT': 0.0,
    'FDTHREADID': 0.0,
    'FDSLICECOUNT1': 0.0,
    'FDCUBEFLAG': 0.0,
    'FDOPERNAME': '',
    'FDF1AQSIGN': 0.0,
    'FDF3AQSIGN': 0.0,
    'FDF4AQSIGN': 0.0,
    'FDF2OFFPPM': 0.0,
    'FDF1OFFPPM': 0.0,
    'FDF3OFFPPM': 0.0,
    'FDF4OFFPPM': 0.0,

    # Extras not reported by show_hdr
    'FDF3FTFLAG': 0.0,
    'FDF3LABEL': 'Z',
    'FDF3OBS': 0.0,
    'FDF3ORIG': 0.0,
    'FDF3SIZE': 1.0,
    'FDF3SW': 0.0,
    'FDF4FTFLAG': 0.0,
    'FDF4LABEL': 'A',
    'FDF4OBS': 0.0,
    'FDF4ORIG': 0.0,
    'FDF4SIZE': 1.0,
    'FDF4SW': 0.0,
    'FDFLTORDER': 2.3450000286102295,
    'FDPLANELOC': 0.0,
    }
}


@pytest.mark.parametrize('in_filepath,meta_answerkey', spectra_exs.items())
def test_load_nmrpipe_meta(in_filepath, meta_answerkey):
    """Test the loading of meta data for NMRPipe files."""
    # Load the meta dict
    with open(in_filepath, 'rb') as f:
        meta = load_nmrpipe_meta(f)

    # Make sure it's the correct type
    assert isinstance(meta, NMRMetaDict)

    # Check the values from the answer key
    for key, value in meta_answerkey.items():
        if isinstance(value, float):
            assert meta[key] == pytest.approx(value, rel=0.001)
        else:
            assert meta[key] == value

    # Make sure there are no missing keys from the meta_answerkey
    missing_keys = meta.keys() - meta_answerkey.keys()
    assert len(missing_keys) == 0


@pytest.mark.parametrize('in_filepath,meta_answerkey', spectra_exs.items())
def test_save_nmrpipe_meta(in_filepath, meta_answerkey):
    """Test the saving of meta data for NMRPipe files."""
    # Load the meta dict
    with open(in_filepath, 'rb') as f:
        meta = load_nmrpipe_meta(f)

    # Save the meta into bytes
    b = save_nmrpipe_meta(meta)

    # Reload the meta from bytes
    meta_saved = load_nmrpipe_meta(io.BytesIO(b))

    # Compare the original with the reloaded
    for k in meta:
        print(k, meta[k], meta_saved[k])
        assert k in meta_saved
        assert meta[k] == meta_saved[k]

