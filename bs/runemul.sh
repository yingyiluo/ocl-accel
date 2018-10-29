aoc -march=emulator device/bs.cl -o bin/bs.aocx
make clean
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host
