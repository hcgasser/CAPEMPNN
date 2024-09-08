#!/usr/bin/env bash

if [ -n "$CAPE" ]; then
    export PF="$CAPE"
fi

export PYTHONPATH="${PF}/libs:${PF}/CAPE-Eval:${PF}/CAPE-XVAE:${PF}/CAPE-Packer:${PF}/CAPE-MPNN"
export PATH="${PF}/CAPE-Eval:${PF}/CAPE-XVAE:${PF}/CAPE-Packer:${PF}/CAPE-MPNN:${PF}/tools:${PATH}"
export LOCH="${PF}/artefacts/CAPE/loch"
