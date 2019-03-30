# Dronet-Carla Benchmarks

Benchmarking [Dronet](https://github.com/uzh-rpg/rpg_public_dronet) on [Carla Simulator](https://github.com/carla-simulator/carla])


```
usage: benchmark.py [-h] [-v] [-db] [--host H] [-p P] [-c C] [-n T]
                    [--corl-2017] [--continue-experiment]
                    [--gpu_fraction GPU_FRACTION]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         print some extra status information
  -db, --debug          print debug information
  --host H              IP of the host server (default: localhost)
  -p P, --port P        TCP port to listen to (default: 2000)
  -c C, --city-name C   The town that is going to be used on benchmark(needs
                        to match active town in server, options: Town01 or
                        Town02)
  -n T, --log_name T    The name of the log file to be created by the
                        benchmark
  --corl-2017           If you want to benchmark the corl-2017 instead of the
                        Basic one
  --continue-experiment
                        If you want to continue the experiment with the same
                        name
  --gpu_fraction GPU_FRACTION
                        Fraction of GPU memory assigned to tensorflow.

```
