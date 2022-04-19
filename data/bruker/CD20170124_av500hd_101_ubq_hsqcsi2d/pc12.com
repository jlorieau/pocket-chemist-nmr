#!/bin/bash

nmrPipe () {
    pc --debug nmrpipe "$@"
    }

#nmrPipe -in test.fid \
#| nmrPipe -fn FT -auto 
# -ov -out out.ft2

nmrPipe -in spec.fid \
| nmrPipe -fn FT -auto \
 -ov -out out.ft2
