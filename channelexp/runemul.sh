aoc -march=emulator device/channeltest.cl -o device/channeltest.aocx
mv device/channeltest.aocx bin/
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host
