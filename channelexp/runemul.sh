aoc -march=emulator device/channeltest.cl -o bin/channeltest.aocx
make clean
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host
