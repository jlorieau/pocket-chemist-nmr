#!/bin/bash

nmrPipe () {
    #pc --debug nmrpipe "$@"
    pc nmrpipe "$@"
    }

#nmrPipe -in test.fid \
#| nmrPipe -fn FT -auto 
# -ov -out out.ft2

nmrPipe -in fid/test%03d.fid \
| nmrPipe -fn FT -auto \
| nmrPipe -out -ov out/out%03d.ft2


