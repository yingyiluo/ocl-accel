aoc -march=emulator -emulator-channel-depth-model=strict device/bs_v17.cl -o bin/bs.aocx
make clean
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/host -v 17
