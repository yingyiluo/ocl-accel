aoc -march=emulator device/bs_v16.cl -o bin/bs.aocx
make clean
make
CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/host
