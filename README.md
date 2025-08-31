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

Performance tested on the Intel Core i7 13700k using a 3840x2160 test image. The numbers indicate that this implementation is up to 12% faster and uses ~40% less memory compared to the [reference implementation](https://github.com/cloudinary/ssimulacra2).

```
poop "ssimulacra2 medium.png dst.png" "./ssimu2 medium.png dst.png"
Benchmark 1 (7 runs): ssimulacra2 medium.png dst.png
  measurement          mean ± σ            min … max           outliers         delta
  wall_time           801ms ± 41.2ms     759ms …  854ms          0 ( 0%)        0%
  peak_rss           1.34GB ± 1.26MB    1.34GB … 1.34GB          0 ( 0%)        0%
  cpu_cycles         3.86G  ±  213M     3.65G  … 4.14G           0 ( 0%)        0%
  instructions       9.33G  ± 3.65M     9.32G  … 9.33G           0 ( 0%)        0%
  cache_references    117M  ± 1.49M      116M  …  119M           0 ( 0%)        0%
  cache_misses       60.6M  ± 2.81M     57.0M  … 63.1M           0 ( 0%)        0%
  branch_misses      16.5M  ± 62.7K     16.4M  … 16.5M           0 ( 0%)        0%
Benchmark 2 (8 runs): ./ssimu2 medium.png dst.png
  measurement          mean ± σ            min … max           outliers         delta
  wall_time           708ms ± 8.95ms     703ms …  730ms          1 (13%)        ⚡- 11.6% ±  4.0%
  peak_rss            817MB ±  127KB     816MB …  817MB          0 ( 0%)        ⚡- 39.0% ±  0.1%
  cpu_cycles         3.55G  ± 25.4M     3.53G  … 3.61G           1 (13%)        ⚡-  8.0% ±  4.2%
  instructions       8.03G  ± 69.7K     8.03G  … 8.03G           1 (13%)        ⚡- 13.9% ±  0.0%
  cache_references   74.7M  ± 8.72K     74.7M  … 74.7M           1 (13%)        ⚡- 36.4% ±  1.0%
  cache_misses       35.9M  ±  103K     35.8M  … 36.1M           0 ( 0%)        ⚡- 40.7% ±  3.5%
  branch_misses      11.5M  ± 16.6K     11.4M  … 11.5M           1 (13%)        ⚡- 30.5% ±  0.3%
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

Compilation requires Zig version 0.15.1

Run `zig build --release=fast`, and the binary will emit to `zig-out/bin/ssimu2`

## License

This project is under the Apache 2.0 license. More details in [LICENSE](./LICENSE).

This project uses code from [libspng](https://libspng.org), [libminiz](https://github.com/richgel999/miniz), and [vapoursynth-zip](https://github.com/dnjulek/vapoursynth-zip). Special thanks to the authors. Licenses for third party code are included in the `legal/` folder.
