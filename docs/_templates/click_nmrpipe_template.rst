.. currentmodule:: {{ module }}

..
    The autodata (auto{{ objtype }}) block is needed to properly link the
    toctree with autosummary. This code block's visibility is set to hidden
    (furo theme css class 'visually-hidden') so that the code block doesn't
    show up on the page.

.. rst-class:: visually-hidden

    .. auto{{ objtype }}:: {{ objname }}

{% set items = name.split('_') %}

.. click:: {{ module }}:{{ name }}
    :prog: pc nmrpipe -{{ items[1] }} {{ items[2] | upper }}
    :nested: short
