# fssimu2

Fast [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2/tree/main) derivative implementation in Zig.

## Usage

```sh
fssimu2 | [version]

usage:
  fssimu2 [--json] reference.png distorted.png

options:
  --json          output result as json
  -h, --help      show this help

8-bit RGB[A] sRGB PNG expected
```

Example output:
```sh
$ ./ssimu2 ref.png dst.png
79.83781132
```

## Performance

Performance tested on the Intel Core i7 13700k using a 3840x2160 test image. The numbers indicate that this implementation is up to 14% faster and uses ~50% less memory.

```
poop "./ssimu2 medium.png dst.png" "ssimulacra2 medium.png dst.png"
Benchmark 1 (8 runs): ./ssimu2 medium.png dst.png
  measurement          mean ± σ            min … max           outliers         delta
  wall_time           701ms ± 3.67ms     699ms …  710ms          1 (13%)        0%
  peak_rss            889MB ±  148KB     889MB …  890MB          1 (13%)        0%
  cpu_cycles         3.49G  ± 5.07M     3.49G  … 3.50G           0 ( 0%)        0%
  instructions       8.21G  ± 78.9K     8.21G  … 8.21G           0 ( 0%)        0%
  cache_references   76.2M  ± 6.30K     76.1M  … 76.2M           0 ( 0%)        0%
  cache_misses       37.0M  ±  204K     36.6M  … 37.3M           0 ( 0%)        0%
  branch_misses      11.8M  ±  235K     11.7M  … 12.4M           1 (13%)        0%
Benchmark 2 (7 runs): ssimulacra2 medium.png dst.png
  measurement          mean ± σ            min … max           outliers         delta
  wall_time           801ms ± 39.7ms     759ms …  854ms          0 ( 0%)        💩+ 14.3% ±  4.3%
  peak_rss           1.34GB ± 1.75MB    1.34GB … 1.34GB          0 ( 0%)        💩+ 50.6% ±  0.2%
  cpu_cycles         3.86G  ±  209M     3.65G  … 4.14G           0 ( 0%)        💩+ 10.4% ±  4.5%
  instructions       9.33G  ± 4.24M     9.32G  … 9.33G           0 ( 0%)        💩+ 13.7% ±  0.0%
  cache_references    118M  ± 1.14M      116M  …  119M           0 ( 0%)        💩+ 54.5% ±  1.1%
  cache_misses       60.6M  ± 2.59M     57.2M  … 63.1M           0 ( 0%)        💩+ 63.9% ±  5.3%
  branch_misses      16.5M  ± 52.0K     16.4M  … 16.6M           0 ( 0%)        💩+ 39.7% ±  1.7%
```

Conformance to the reference SSIMULACRA2 implementation can be tested with `validate.py` by supplying the `ssimu2` binary.

`validate.py` requires [`uv`](https://docs.astral.sh/uv/) and [`libjxl`](https://github.com/libjxl/libjxl).

```sh
validate.py --custom ~/git-cloning/fssimu2/zig-out/bin/ssimu2 ~/git-cloning/gb82-image-set/png/*
```

Output on the [gb82 image set](https://github.com/gianni-rosato/gb82-image-set) invoking the above command:
```
SAMPLES: 75
LEVELS: 1.0 2.0 4.0

== custom vs ref ==
 pairs: 75
 ref mean: 77.5116  std: 10.6899
 custom mean: 76.9932  std: 11.1606
 mean diff (other - ref): -0.518408
 diff stddev: 0.541577
 diff stderr: 0.0625359
 percentage error (mean diff / ref mean): 0.669%
 max absolute error: 1.9165

== correlation ==
 PCC (Pearson): 0.999700
 SRCC (Spearman): 0.999403
 KRCC (Kendall): 0.987748
```

## Compilation

Compilation requires Zig version 0.14.1

Run `zig build --release=fast`, and the binary will emit to `zig-out/bin/ssimu2`

## License

This project is under the Apache 2.0 license. More details in [LICENSE](./LICENSE).

This project uses code from [libspng](https://libspng.org), [libminiz](https://github.com/richgel999/miniz), and [vapoursynth-zip](https://github.com/dnjulek/vapoursynth-zip). Special thanks to the authors. Licenses for third party code are included in the `legal/` folder.
