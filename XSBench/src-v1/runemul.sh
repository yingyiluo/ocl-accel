aoc -march=emulator -emulator-channel-depth-model=strict device/Simulation.cl -o bin/Simulation.aocx
make clean
make
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./bin/XSBench
