NMRSpectrum
===========
.. automodule:: pocketchemist_nmr.spectra

.. currentmodule:: pocketchemist_nmr.spectra

.. autoclass:: NMRSpectrum

Attributes
----------

.. autoattribute:: NMRSpectrum.meta
.. autoattribute:: NMRSpectrum.data
.. autoattribute:: NMRSpectrum.in_filepath
.. autoattribute:: NMRSpectrum.out_filepath

Accessors
---------

.. autoproperty:: NMRSpectrum.ndims
.. autoproperty:: NMRSpectrum.npts
.. autoproperty:: NMRSpectrum.npts_data
.. autoproperty:: NMRSpectrum.domain_type
.. autoproperty:: NMRSpectrum.data_type
.. autoproperty:: NMRSpectrum.sw_hz
.. autoproperty:: NMRSpectrum.sw_ppm
.. autoproperty:: NMRSpectrum.car_hz
.. autoproperty:: NMRSpectrum.car_ppm
.. autoproperty:: NMRSpectrum.obs_mhz
.. autoproperty:: NMRSpectrum.range_hz
.. autoproperty:: NMRSpectrum.range_ppm
.. autoproperty:: NMRSpectrum.range_s
.. autoproperty:: NMRSpectrum.array_hz
.. autoproperty:: NMRSpectrum.array_ppm
.. autoproperty:: NMRSpectrum.array_s
.. autoproperty:: NMRSpectrum.label
.. autoproperty:: NMRSpectrum.apodization
.. autoproperty:: NMRSpectrum.data_layout
.. autoproperty:: NMRSpectrum.group_delay
.. autoproperty:: NMRSpectrum.correct_digital_filter
.. automethod:: NMRSpectrum.data_layout
.. automethod:: NMRSpectrum.convert

Saving and Loading
------------------

.. automethod:: NMRSpectrum.save
.. automethod:: NMRSpectrum.load

Processing
----------

.. autosummary::
    :toctree: processing
    :template: methodtemplate.rst
    :nosignatures:

    NMRSpectrum.apodization_exp
    NMRSpectrum.apodization_sine
    NMRSpectrum.extract
    NMRSpectrum.ft
    NMRSpectrum.transpose
    NMRSpectrum.phase
    NMRSpectrum.zerofill