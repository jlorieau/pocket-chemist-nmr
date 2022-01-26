"""
Functions to parse NMRPipe headers
"""
import typing as t

from ..meta import NMRMetaDescriptionDict

__all__ = ('get_nmrpipe_definitions',)

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

    # Copied from NMRPipe/nmruser/fdatap.h
    # TODO: The following regexes should be replaced by a proper C header parser
    offsets_str = """
/***/
/* General Parameter locations:
/***/

#define FDMAGIC        0 /* Should be zero in valid NMRPipe data.            */
#define FDFLTFORMAT    1 /* Constant defining floating point format.         */
#define FDFLTORDER     2 /* Constant defining byte order.                    */

#define FDSIZE        99 /* Number of points in current dim R|I.             */
#define FDREALSIZE    97 /* Number of valid time-domain pts (obsolete).      */
#define FDSPECNUM    219 /* Number of complex 1D slices in file.             */
#define FDQUADFLAG   106 /* See Data Type codes below.                       */
#define FD2DPHASE    256 /* See 2D Plane Type codes below.                   */


/***/
/* Parameters defining number of dimensions and their order in the data;
/* a newly-converted FID has dimension order (2 1 3 4). These dimension
/* codes are a hold-over from the oldest 2D NMR definitions, where the
/* directly-acquired dimension was always t2, and the indirect dimension
/* was t1.
/***/

#define FDTRANSPOSED 221 /* 1=Transposed, 0=Not Transposed.                  */
#define FDDIMCOUNT     9 /* Number of dimensions in complete data.           */
#define FDDIMORDER    24 /* Array describing dimension order.                */

#define FDDIMORDER1   24 /* Dimension stored in X-Axis.                      */
#define FDDIMORDER2   25 /* Dimension stored in Y-Axis.                      */
#define FDDIMORDER3   26 /* Dimension stored in Z-Axis.                      */
#define FDDIMORDER4   27 /* Dimension stored in A-Axis.                      */

#define FDNUSDIM      45 /* Unexpanded NUS dimensions.                       */

/***/
/* The following parameters describe the data when it is
/* in a multidimensional data stream format (FDPIPEFLAG != 0).
/* To accommodate large data, total number of 1D vectors
/* is described as:
/*
/*   fdata[FDSLICECOUNT0] + MAX_NMR_SIZE*fdata[FDSLICECOUNT1]
/***/

#define FDPIPEFLAG     57 /* Dimension code of data stream.                  */
#define FDCUBEFLAG    447 /* Data is 3D cube file series.                    */
#define FDPIPECOUNT    75 /* Number of functions in pipe.                    */
#define FDSLICECOUNT0 443 /* Encodes number of 1D slices in stream.          */
#define FDSLICECOUNT1 446 /* Encodes number of 1D slices in stream.          */
#define FDFILECOUNT   442 /* Number of files in complete data.               */

#define FDTHREADCOUNT 444 /* Multi-Thread Mode: Number of Threads.           */
#define FDTHREADID    445 /* Multi-Thread Mode: Thread ID, First = 0.        */

/***/
/* The following definitions are used for data streams which are
/* subsets of the complete data, as for parallel processing:
/***/

#define FDFIRSTPLANE  77 /* First Z-Plane in subset.       Added for NMRPipe. */
#define FDLASTPLANE   78 /* Last Z-Plane in subset.        Added for NMRPipe. */
#define FDPARTITION   65 /* Slice count for server mode.   Added for NMRPipe. */

#define FDPLANELOC    14 /* Location of this plane; currently unused.         */

/***/
/* The following define max and min data values, previously used
/* for contour level setting:
/***/

#define FDMAX        247 /* Max value in real part of data.                  */
#define FDMIN        248 /* Min value in real part of data.                  */
#define FDSCALEFLAG  250 /* 1 if FDMAX and FDMIN are valid.                  */
#define FDDISPMAX    251 /* Max value, used for display generation.          */
#define FDDISPMIN    252 /* Min value, used for display generation.          */
#define FDPTHRESH    253 /* Positive threshold for peak detection.           */
#define FDNTHRESH    254 /* Negative threshold for peak detection.           */

/***/
/* Locations reserved for User customization:
/***/

#define FDUSER1       70
#define FDUSER2       71
#define FDUSER3       72
#define FDUSER4       73
#define FDUSER5       74
#define FDUSER6       76

/***/
/* Defines location of "footer" information appended to spectral
/* data; currently unused for NMRPipe format:
/***/

#define FDLASTBLOCK  359
#define FDCONTBLOCK  360
#define FDBASEBLOCK  361
#define FDPEAKBLOCK  362
#define FDBMAPBLOCK  363
#define FDHISTBLOCK  364
#define FD1DBLOCK    365

/***/
/* Defines data and time data was converted:
/***/

#define FDMONTH      294
#define FDDAY        295
#define FDYEAR       296
#define FDHOURS      283
#define FDMINS       284
#define FDSECS       285

/***/
/* Miscellaneous Parameters:
/***/

#define FDMCFLAG      135 /* Magnitude Calculation performed.               */
#define FDNOISE       153 /* Used to contain an RMS noise estimate.         */
#define FDRANK        180 /* Estimate of matrix rank; Added for NMRPipe.    */
#define FDTEMPERATURE 157 /* Temperature, K.                                */
#define FDPRESSURE    158 /* Pressure, Pascal.                              */
#define FD2DVIRGIN    399 /* 0=Data never accessed, header never adjusted.  */
#define FDTAU         199 /* A Tau value (for spectral series).             */
#define FDDOMINFO     266 /* Spectral/Spatial Flags. Added for NMRPipe.     */
#define FDMETHINFO    267 /* FT/Direct Flags. Added for NMRPipe.            */

#define FDSCORE       370 /* Added for screening score etc.                 */
#define FDSCANS       371 /* Number of Scans per 1D.                        */

#define FDSRCNAME    286  /* char srcFile[16]  286-289 */
#define FDUSERNAME   290  /* char uName[16]    290-293 */
#define FDOPERNAME   464  /* char oName[32]    464-471 */
#define FDTITLE      297  /* char title[60]    297-311 */
#define FDCOMMENT    312  /* char comment[160] 312-351 */

/***/
/* For meanings of these dimension-specific parameters,
/* see the corresponding ND parameters below.
/***/

#define FDF2LABEL     16
#define FDF2APOD      95
#define FDF2SW       100
#define FDF2OBS      119
#define FDF2OBSMID   378
#define FDF2ORIG     101
#define FDF2UNITS    152
#define FDF2QUADFLAG  56 /* Added for NMRPipe. */
#define FDF2FTFLAG   220
#define FDF2AQSIGN    64 /* Added for NMRPipe. */
#define FDF2CAR       66 /* Added for NMRPipe. */
#define FDF2CENTER    79 /* Added for NMRPipe. */
#define FDF2OFFPPM   480 /* Added for NMRPipe. */
#define FDF2P0       109
#define FDF2P1       110
#define FDF2APODCODE 413
#define FDF2APODQ1   415
#define FDF2APODQ2   416
#define FDF2APODQ3   417
#define FDF2LB       111
#define FDF2GB       374
#define FDF2GOFF     382
#define FDF2C1       418
#define FDF2APODDF   419
#define FDF2ZF       108
#define FDF2X1       257 /* Added for NMRPipe. */
#define FDF2XN       258 /* Added for NMRPipe. */
#define FDF2FTSIZE    96 /* Added for NMRPipe. */
#define FDF2TDSIZE   386 /* Added for NMRPipe. */

#define FDDMXVAL      40 /* Added for NMRPipe. */
#define FDDMXFLAG     41 /* Added for NMRPipe. */
#define FDDELTATR     42 /* Added for NMRPipe. */

#define FDF1LABEL     18
#define FDF1APOD     428
#define FDF1SW       229 
#define FDF1OBS      218 
#define FDF1OBSMID   379
#define FDF1ORIG     249 
#define FDF1UNITS    234 
#define FDF1FTFLAG   222 
#define FDF1AQSIGN   475 /* Added for NMRPipe. */
#define FDF1QUADFLAG  55 /* Added for NMRPipe. */
#define FDF1CAR       67 /* Added for NMRPipe. */
#define FDF1CENTER    80 /* Added for NMRPipe. */
#define FDF1OFFPPM   481 /* Added for NMRPipe. */
#define FDF1P0       245
#define FDF1P1       246
#define FDF1APODCODE 414
#define FDF1APODQ1   420
#define FDF1APODQ2   421 
#define FDF1APODQ3   422
#define FDF1LB       243
#define FDF1GB       375
#define FDF1GOFF     383
#define FDF1C1       423
#define FDF1ZF       437
#define FDF1X1       259 /* Added for NMRPipe. */
#define FDF1XN       260 /* Added for NMRPipe. */
#define FDF1FTSIZE    98 /* Added for NMRPipe. */
#define FDF1TDSIZE   387 /* Added for NMRPipe. */

#define FDF3LABEL     20
#define FDF3APOD      50 /* Added for NMRPipe. */
#define FDF3OBS       10
#define FDF3OBSMID   380
#define FDF3SW        11
#define FDF3ORIG      12
#define FDF3FTFLAG    13
#define FDF3AQSIGN   476 /* Added for NMRPipe. */
#define FDF3SIZE      15
#define FDF3QUADFLAG  51 /* Added for NMRPipe. */
#define FDF3UNITS     58 /* Added for NMRPipe. */
#define FDF3P0        60 /* Added for NMRPipe. */
#define FDF3P1        61 /* Added for NMRPipe. */
#define FDF3CAR       68 /* Added for NMRPipe. */
#define FDF3CENTER    81 /* Added for NMRPipe. */
#define FDF3OFFPPM   482 /* Added for NMRPipe. */
#define FDF3APODCODE 400 /* Added for NMRPipe. */
#define FDF3APODQ1   401 /* Added for NMRPipe. */
#define FDF3APODQ2   402 /* Added for NMRPipe. */
#define FDF3APODQ3   403 /* Added for NMRPipe. */
#define FDF3LB       372 /* Added for NMRPipe. */
#define FDF3GB       376 /* Added for NMRPipe. */
#define FDF3GOFF     384 /* Added for NMRPipe. */
#define FDF3C1       404 /* Added for NMRPipe. */
#define FDF3ZF       438 /* Added for NMRPipe. */
#define FDF3X1       261 /* Added for NMRPipe. */
#define FDF3XN       262 /* Added for NMRPipe. */
#define FDF3FTSIZE   200 /* Added for NMRPipe. */
#define FDF3TDSIZE   388 /* Added for NMRPipe. */

#define FDF4LABEL     22
#define FDF4APOD      53 /* Added for NMRPipe. */
#define FDF4OBS       28
#define FDF4OBSMID   381
#define FDF4SW        29
#define FDF4ORIG      30
#define FDF4FTFLAG    31
#define FDF4AQSIGN   477 /* Added for NMRPipe. */
#define FDF4SIZE      32
#define FDF4QUADFLAG  54 /* Added for NMRPipe. */
#define FDF4UNITS     59 /* Added for NMRPipe. */
#define FDF4P0        62 /* Added for NMRPipe. */
#define FDF4P1        63 /* Added for NMRPipe. */
#define FDF4CAR       69 /* Added for NMRPipe. */
#define FDF4CENTER    82 /* Added for NMRPipe. */
#define FDF4OFFPPM   483 /* Added for NMRPipe. */
#define FDF4APODCODE 405 /* Added for NMRPipe. */
#define FDF4APODQ1   406 /* Added for NMRPipe. */
#define FDF4APODQ2   407 /* Added for NMRPipe. */
#define FDF4APODQ3   408 /* Added for NMRPipe. */
#define FDF4LB       373 /* Added for NMRPipe. */
#define FDF4GB       377 /* Added for NMRPipe. */
#define FDF4GOFF     385 /* Added for NMRPipe. */
#define FDF4C1       409 /* Added for NMRPipe. */
#define FDF4ZF       439 /* Added for NMRPipe. */
#define FDF4X1       263 /* Added for NMRPipe. */
#define FDF4XN       264 /* Added for NMRPipe. */
#define FDF4FTSIZE   201 /* Added for NMRPipe. */
#define FDF4TDSIZE   389 /* Added for NMRPipe. */
"""
    # Prepare the fields that should be converted to strings
    text_fields_str = """
#define SIZE_NDLABEL    8
#define SIZE_F2LABEL    8
#define SIZE_F1LABEL    8
#define SIZE_F3LABEL    8
#define SIZE_F4LABEL    8

#define SIZE_SRCNAME   16
#define SIZE_USERNAME  16
#define SIZE_OPERNAME  32
#define SIZE_COMMENT  160
#define SIZE_TITLE     60
"""
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
