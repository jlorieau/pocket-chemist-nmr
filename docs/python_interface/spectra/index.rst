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
.. autoproperty:: NMRSpectrum.domain_type
.. autoproperty:: NMRSpectrum.data_type
.. autoproperty:: NMRSpectrum.sw
.. autoproperty:: NMRSpectrum.label
.. autoproperty:: NMRSpectrum.apodization
.. autoproperty:: NMRSpectrum.npts
.. autoproperty:: NMRSpectrum.data_layout
.. autoproperty:: NMRSpectrum.group_delay
.. autoproperty:: NMRSpectrum.correct_digital_filter

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
    NMRSpectrum.ft
    NMRSpectrum.transpose
    NMRSpectrum.phase