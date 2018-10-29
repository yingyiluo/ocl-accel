aoc -march=emulator device/bs.cl -o device/bs.aocx
mv device/bs.aocx bin/
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host
