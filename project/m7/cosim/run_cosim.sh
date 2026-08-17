#!/bin/bash
# Full-dataset RTL cosim under the matched x86_64 cocotb + iverilog toolchain.
cd "$(dirname "$0")"
rm -rf sim_build results.xml
arch -x86_64 /bin/bash -c '
  export PATH="/tmp/x86cocotb/bin:/usr/local/bin:$PATH"
  make
'
