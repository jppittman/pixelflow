# Class cap: live-counted budget A/B (2026-09-02)

Corpus: 206 real kernels + 200 synthetic = 406 pairs. Both arms in one process through `Optimizer::production()`; only the budget differs. Cost is `CostModel::latency_prior()` over the extracted arena. Load average 5.17 -> 5.03 — **LOADED**, so every wall-clock number here is context, not a measurement.

- **before (`allocated`)**: `Budget::Explicit { classes: HARD_CLASS_LIMIT, allocated_classes: preset.max_classes }` — every budget check reduces to the old `self.classes.len() > max_classes`.
- **after (`live`)**: `Budget::Production` — the preset's `max_classes` as the LIVE budget, `HARD_CLASS_LIMIT` (100 000) as the allocated memory guard.

## Headline

- ALL: ALL (n=406): cost median +0.000% p90 +4.603% mean +1.398% | improved 158 regressed 10 | live median 420->562 | allocated median 2884->4724 (worst 2.57x) | bytes-proxy median 495040->807492 (worst 2.58x) | applications median 1.54x p90 2.80x | wall median 1.24x p90 3.58x | capped 231->226 (allocated-guard 0)
- REAL (all non-synthetic) (n=206): cost median +2.026% p90 +6.270% mean +2.742% | improved 155 regressed 4 | live median 1379->2172 | allocated median 4688->7220 (worst 2.57x) | bytes-proxy median 797526->1216040 (worst 2.58x) | applications median 2.00x p90 3.35x | wall median 1.47x p90 4.37x | capped 191->189 (allocated-guard 0)

## Per group

| group | n | cost median | cost p90 | improved | regressed | live median | allocated median | alloc worst | bytes-proxy median | bytes worst | apps median | wall median | capped before→after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glyph16 | 95 | +2.09% | +6.27% | 75 | 1 | 1703→2453 | 4718→7225 | 2.14x | 804048→1219120 | 2.15x | 2.03x | 1.47x | 89→88 |
| glyph32 | 95 | +2.09% | +6.27% | 75 | 1 | 1703→2453 | 4718→7225 | 2.14x | 804048→1219120 | 2.15x | 2.03x | 1.49x | 89→88 |
| shader | 12 | +0.00% | +8.82% | 5 | 1 | 311→406 | 1794→3648 | 2.57x | 303548→617820 | 2.58x | 1.61x | 1.35x | 9→9 |
| psychedelic | 1 | -5.35% | -5.35% | 0 | 1 | 946→2561 | 4847→9885 | 2.04x | 822528→1669584 | 2.03x | 1.90x | 1.77x | 1→1 |
| cellgrid | 3 | +0.00% | +0.00% | 0 | 0 | 1261→1763 | 4583→8034 | 1.75x | 778512→1361360 | 1.75x | 1.67x | 1.42x | 3→3 |
| synthetic | 200 | +0.00% | +0.00% | 3 | 6 | 67→67 | 161→161 | 2.51x | 27496→27496 | 2.55x | 1.00x | 0.98x | 40→37 |
| REAL (all non-synthetic) | 206 | +2.03% | +6.27% | 155 | 4 | 1379→2172 | 4688→7220 | 2.57x | 797526→1216040 | 2.58x | 2.00x | 1.47x | 191→189 |
| REAL cap-hit (before) | 191 | +2.14% | +6.45% | 155 | 4 | 1703→2453 | 4718→7298 | 2.57x | 804048→1230096 | 2.58x | 2.06x | 1.49x | 191→189 |
| ALL | 406 | +0.00% | +4.60% | 158 | 10 | 420→562 | 2884→4724 | 2.57x | 495040→807492 | 2.58x | 1.54x | 1.24x | 231→226 |

## Verdict

- **Quality**: real kernels move +2.03% median / +6.27% p90 cheaper; 155 of 206 improve, 4 regress.
- **Memory**: the largest graph any kernel allocates goes to 10660 classes (1.83 MB by the byte proxy), against the `HARD_CLASS_LIMIT` guard of 100000. The guard fired on 0 kernels, so the allocated ceiling is not what bounds this corpus — the live budget is.
- **Compile cost**: rule applications, the deterministic proxy, go 2.00x median / 3.35x p90 on real kernels. Aggregate wall clock over the real corpus is 7.78 s -> 23.71 s (3.05x) — context only, the machine was loaded.


The trade is real but not free, and it is not a pure win: a couple of percent of extracted cost on real kernels for roughly double the saturation work and double the peak e-graph. The regressions below are why this is a judgement call rather than an obvious yes — a bigger e-graph does not monotonically produce a cheaper extraction, because extraction is a greedy DP over a static cost prior, not an optimum, so more equalities can move the greedy choice onto a worse branch.

## Cost regressions on real kernels

| kernel | group | before | after | delta |
|---|---|---|---|---|
| glyph16:U+004B | glyph16 | 1047 | 1121 | +74 |
| glyph32:U+004B | glyph32 | 1047 | 1121 | +74 |
| psychedelic | psychedelic | 766 | 807 | +41 |
| shader:julia_set | shader | 716 | 728 | +12 |

## Per kernel

| kernel | group | nodes | stop before → after | live before → after | allocated before → after | bytes-proxy before → after | apps before → after | cost before → after | improvement |
|---|---|---|---|---|---|---|---|---|---|
| glyph16:U+0027 | glyph16 | 127 | ClassCap(Allocated) → ClassCap(Live) | 582 → 1240 | 3566 → 7642 | 609896 → 1297632 | 16399 → 32181 | 171 → 137 | +19.88% |
| glyph32:U+0027 | glyph32 | 127 | ClassCap(Allocated) → ClassCap(Live) | 582 → 1240 | 3566 → 7642 | 609896 → 1297632 | 16399 → 32181 | 171 → 137 | +19.88% |
| glyph16:U+002F | glyph16 | 127 | ClassCap(Allocated) → ClassCap(Live) | 447 → 653 | 3444 → 6235 | 589236 → 1062716 | 18357 → 38385 | 166 → 138 | +16.87% |
| glyph16:U+005C | glyph16 | 127 | ClassCap(Allocated) → ClassCap(Live) | 447 → 653 | 3444 → 6235 | 589236 → 1062716 | 18357 → 38385 | 166 → 138 | +16.87% |
| glyph32:U+002F | glyph32 | 127 | ClassCap(Allocated) → ClassCap(Live) | 447 → 653 | 3444 → 6235 | 589236 → 1062716 | 18357 → 38385 | 166 → 138 | +16.87% |
| glyph32:U+005C | glyph32 | 127 | ClassCap(Allocated) → ClassCap(Live) | 447 → 653 | 3444 → 6235 | 589236 → 1062716 | 18357 → 38385 | 166 → 138 | +16.87% |
| shader:metaballs | shader | 62 | ClassCap(Allocated) → ClassCap(Live) | 286 → 590 | 2584 → 4575 | 450800 → 787696 | 10090 → 20425 | 158 → 135 | +14.56% |
| glyph16:U+004A | glyph16 | 2425 | ClassCap(Allocated) → ClassCap(Live) | 1868 → 2490 | 4282 → 8998 | 720496 → 1526056 | 3220 → 13711 | 4339 → 3926 | +9.52% |
| glyph32:U+004A | glyph32 | 2425 | ClassCap(Allocated) → ClassCap(Live) | 1868 → 2490 | 4282 → 8998 | 720496 → 1526056 | 3220 → 13711 | 4339 → 3926 | +9.52% |
| glyph16:U+0055 | glyph16 | 2899 | ClassCap(Allocated) → ClassCap(Live) | 2079 → 2732 | 4686 → 9723 | 788032 → 1647520 | 3440 → 14671 | 5170 → 4701 | +9.07% |
| glyph32:U+0055 | glyph32 | 2899 | ClassCap(Allocated) → ClassCap(Live) | 2079 → 2732 | 4686 → 9723 | 788032 → 1647520 | 3440 → 14671 | 5170 → 4701 | +9.07% |
| shader:smooth_min_scene | shader | 43 | ClassCap(Allocated) → ClassCap(Live) | 160 → 170 | 1219 → 1591 | 213808 → 276976 | 2722 → 6504 | 136 → 124 | +8.82% |
| glyph16:U+003C | glyph16 | 295 | ClassCap(Allocated) → ClassCap(Live) | 420 → 435 | 2697 → 3406 | 468328 → 588000 | 7887 → 24555 | 514 → 470 | +8.56% |
| glyph32:U+003C | glyph32 | 295 | ClassCap(Allocated) → ClassCap(Live) | 420 → 436 | 2697 → 3407 | 468328 → 588168 | 7887 → 24555 | 514 → 470 | +8.56% |
| glyph16:U+003E | glyph16 | 295 | ClassCap(Allocated) → ClassCap(Live) | 421 → 436 | 2711 → 3419 | 470512 → 590016 | 7897 → 24643 | 519 → 475 | +8.48% |
| glyph32:U+003E | glyph32 | 295 | ClassCap(Allocated) → ClassCap(Live) | 421 → 437 | 2711 → 3420 | 470512 → 590184 | 7897 → 24641 | 519 → 475 | +8.48% |
| glyph16:U+003B | glyph16 | 3494 | ClassCap(Allocated) → ClassCap(Live) | 2307 → 3029 | 4855 → 10367 | 816816 → 1757112 | 3371 → 15562 | 6080 → 5609 | +7.75% |
| glyph32:U+003B | glyph32 | 3494 | ClassCap(Allocated) → ClassCap(Live) | 2307 → 3029 | 4855 → 10367 | 816816 → 1757112 | 3371 → 15562 | 6080 → 5609 | +7.75% |
| synth_d11_s23 | synthetic | 221 | ClassCap(Allocated) → ClassCap(Live) | 976 → 2348 | 4247 → 10660 | 719880 → 1832880 | 10168 → 24798 | 2569 → 2403 | +6.46% |
| glyph16:U+0022 | glyph16 | 199 | ClassCap(Allocated) → ClassCap(Live) | 421 → 738 | 2888 → 5309 | 497728 → 906416 | 12044 → 23647 | 310 → 290 | +6.45% |
| glyph32:U+0022 | glyph32 | 199 | ClassCap(Allocated) → ClassCap(Live) | 421 → 738 | 2888 → 5309 | 497728 → 906416 | 12044 → 23647 | 310 → 290 | +6.45% |
| glyph16:U+005E | glyph16 | 199 | ClassCap(Allocated) → ClassCap(Live) | 420 → 717 | 2891 → 5184 | 498288 → 885360 | 12176 → 23737 | 319 → 299 | +6.27% |
| glyph32:U+005E | glyph32 | 199 | ClassCap(Allocated) → ClassCap(Live) | 420 → 717 | 2891 → 5184 | 498288 → 885360 | 12176 → 23737 | 319 → 299 | +6.27% |
| glyph16:U+0078 | glyph16 | 343 | ClassCap(Allocated) → ClassCap(Live) | 620 → 805 | 3590 → 5695 | 621488 → 979496 | 8967 → 22114 | 639 → 599 | +6.26% |
| glyph32:U+0078 | glyph32 | 343 | ClassCap(Allocated) → ClassCap(Live) | 620 → 805 | 3590 → 5695 | 621488 → 979496 | 8967 → 22114 | 639 → 599 | +6.26% |
| glyph16:U+0058 | glyph16 | 343 | ClassCap(Allocated) → ClassCap(Live) | 628 → 805 | 3607 → 5659 | 624456 → 973392 | 9017 → 22137 | 652 → 612 | +6.13% |
| glyph32:U+0058 | glyph32 | 343 | ClassCap(Allocated) → ClassCap(Live) | 628 → 805 | 3607 → 5659 | 624456 → 973392 | 9017 → 22137 | 652 → 612 | +6.13% |
| glyph16:U+0044 | glyph16 | 2606 | ClassCap(Allocated) → ClassCap(Live) | 1933 → 2572 | 4367 → 9105 | 734944 → 1544256 | 3265 → 13737 | 4506 → 4230 | +6.13% |
| glyph32:U+0044 | glyph32 | 2606 | ClassCap(Allocated) → ClassCap(Live) | 1933 → 2572 | 4367 → 9105 | 734944 → 1544256 | 3265 → 13737 | 4506 → 4230 | +6.13% |
| glyph16:U+0052 | glyph16 | 2710 | ClassCap(Allocated) → ClassCap(Live) | 2000 → 2632 | 4509 → 9321 | 758520 → 1582000 | 3337 → 14111 | 4725 → 4443 | +5.97% |
| glyph32:U+0052 | glyph32 | 2710 | ClassCap(Allocated) → ClassCap(Live) | 2000 → 2632 | 4509 → 9321 | 758520 → 1582000 | 3337 → 14111 | 4725 → 4443 | +5.97% |
| glyph16:U+0068 | glyph16 | 2931 | ClassCap(Allocated) → ClassCap(Live) | 2082 → 2744 | 4717 → 9818 | 793800 → 1666000 | 3472 → 14780 | 5045 → 4751 | +5.83% |
| glyph32:U+0068 | glyph32 | 2931 | ClassCap(Allocated) → ClassCap(Live) | 2082 → 2744 | 4717 → 9818 | 793800 → 1666000 | 3472 → 14780 | 5045 → 4751 | +5.83% |
| glyph16:U+0059 | glyph16 | 263 | ClassCap(Allocated) → ClassCap(Live) | 412 → 556 | 2684 → 4061 | 465360 → 698376 | 7570 → 16991 | 437 → 413 | +5.49% |
| glyph32:U+0059 | glyph32 | 263 | ClassCap(Allocated) → ClassCap(Live) | 412 → 555 | 2684 → 4060 | 465360 → 698208 | 7570 → 16991 | 437 → 413 | +5.49% |
| glyph16:U+006D | glyph16 | 5289 | ClassCap(Allocated) → ClassCap(Live) | 2856 → 3671 | 4998 → 8013 | 843248 → 1349768 | 2654 → 5680 | 9624 → 9138 | +5.05% |
| glyph32:U+006D | glyph32 | 5289 | ClassCap(Allocated) → ClassCap(Live) | 2856 → 3671 | 4998 → 8013 | 843248 → 1349768 | 2654 → 5680 | 9624 → 9138 | +5.05% |
| glyph16:U+0076 | glyph16 | 1115 | ClassCap(Allocated) → ClassCap(Live) | 1182 → 1673 | 4236 → 8688 | 721728 → 1487136 | 6760 → 20494 | 1760 → 1676 | +4.77% |
| glyph32:U+0076 | glyph32 | 1115 | ClassCap(Allocated) → ClassCap(Live) | 1182 → 1673 | 4236 → 8688 | 721728 → 1487136 | 6760 → 20494 | 1760 → 1676 | +4.77% |
| glyph16:U+0041 | glyph16 | 1605 | ClassCap(Allocated) → ClassCap(Live) | 1340 → 1854 | 4509 → 6470 | 770112 → 1100288 | 6637 → 9948 | 2672 → 2549 | +4.60% |
| glyph32:U+0041 | glyph32 | 1605 | ClassCap(Allocated) → ClassCap(Live) | 1340 → 1854 | 4509 → 6470 | 770112 → 1100288 | 6637 → 9948 | 2672 → 2549 | +4.60% |
| glyph16:U+006B | glyph16 | 1460 | ClassCap(Allocated) → ClassCap(Live) | 1306 → 1773 | 4460 → 6279 | 761992 → 1068200 | 6789 → 9796 | 2517 → 2404 | +4.49% |
| glyph32:U+006B | glyph32 | 1460 | ClassCap(Allocated) → ClassCap(Live) | 1306 → 1773 | 4460 → 6279 | 761992 → 1068200 | 6789 → 9796 | 2517 → 2404 | +4.49% |
| glyph16:U+004E | glyph16 | 1399 | ClassCap(Allocated) → ClassCap(Live) | 1325 → 2010 | 4653 → 7495 | 796040 → 1274616 | 6421 → 11445 | 2457 → 2348 | +4.44% |
| glyph32:U+004E | glyph32 | 1399 | ClassCap(Allocated) → ClassCap(Live) | 1325 → 2010 | 4653 → 7495 | 796040 → 1274616 | 6421 → 11445 | 2457 → 2348 | +4.44% |
| glyph16:U+0034 | glyph16 | 1637 | ClassCap(Allocated) → ClassCap(Live) | 1337 → 1888 | 4557 → 6693 | 778848 → 1138536 | 6666 → 10207 | 2751 → 2630 | +4.40% |
| glyph32:U+0034 | glyph32 | 1637 | ClassCap(Allocated) → ClassCap(Live) | 1337 → 1888 | 4557 → 6693 | 778848 → 1138536 | 6666 → 10207 | 2751 → 2630 | +4.40% |
| glyph16:U+0056 | glyph16 | 1573 | ClassCap(Allocated) → ClassCap(Live) | 1319 → 1920 | 4614 → 6922 | 787080 → 1176840 | 6581 → 10677 | 2647 → 2534 | +4.27% |
| glyph32:U+0056 | glyph32 | 1573 | ClassCap(Allocated) → ClassCap(Live) | 1319 → 1920 | 4614 → 6922 | 787080 → 1176840 | 6581 → 10677 | 2647 → 2534 | +4.27% |
| glyph16:U+0031 | glyph16 | 1452 | ClassCap(Allocated) → ClassCap(Live) | 1323 → 1724 | 4418 → 6156 | 755440 → 1048152 | 6800 → 9393 | 2520 → 2413 | +4.25% |
| glyph32:U+0031 | glyph32 | 1452 | ClassCap(Allocated) → ClassCap(Live) | 1323 → 1724 | 4418 → 6156 | 755440 → 1048152 | 6800 → 9393 | 2520 → 2413 | +4.25% |
| shader:domain_warp_fbm | shader | 84 | ClassCap(Allocated) → ClassCap(Live) | 1245 → 1908 | 4577 → 7885 | 778512 → 1337280 | 6976 → 11236 | 444 → 426 | +4.05% |
| glyph16:U+0049 | glyph16 | 391 | ClassCap(Allocated) → ClassCap(Live) | 565 → 695 | 3418 → 5001 | 591248 → 859264 | 8929 → 20753 | 626 → 602 | +3.83% |
| glyph32:U+0049 | glyph32 | 391 | ClassCap(Allocated) → ClassCap(Live) | 565 → 694 | 3418 → 5000 | 591248 → 859096 | 8929 → 20755 | 626 → 602 | +3.83% |
| glyph16:U+006C | glyph16 | 323 | ClassCap(Allocated) → ClassCap(Live) | 467 → 591 | 3007 → 4242 | 520576 → 729232 | 8304 → 18346 | 522 → 503 | +3.64% |
| glyph32:U+006C | glyph32 | 323 | ClassCap(Allocated) → ClassCap(Live) | 467 → 592 | 3007 → 4243 | 520576 → 729400 | 8304 → 18346 | 522 → 503 | +3.64% |
| glyph16:U+0037 | glyph16 | 191 | ClassCap(Allocated) → ClassCap(Live) | 378 → 604 | 2792 → 4931 | 481096 → 844032 | 12362 → 24454 | 288 → 278 | +3.47% |
| glyph32:U+0037 | glyph32 | 191 | ClassCap(Allocated) → ClassCap(Live) | 377 → 603 | 2791 → 4930 | 480928 → 843864 | 12363 → 24455 | 288 → 278 | +3.47% |
| glyph16:U+0039 | glyph16 | 8267 | ClassCap(Allocated) → ClassCap(Live) | 3774 → 4626 | 5000 → 8215 | 842016 → 1384432 | 1260 → 4482 | 15698 → 15201 | +3.17% |
| glyph32:U+0039 | glyph32 | 8267 | ClassCap(Allocated) → ClassCap(Live) | 3774 → 4626 | 5000 → 8215 | 842016 → 1384432 | 1260 → 4482 | 15698 → 15201 | +3.17% |
| shader:mandelbrot_distance | shader | 152 | ClassCap(Allocated) → ClassCap(Live) | 415 → 299 | 3678 → 5718 | 632184 → 987952 | 15291 → 33236 | 576 → 558 | +3.12% |
| glyph16:U+0042 | glyph16 | 4470 | ClassCap(Allocated) → ClassCap(Live) | 2650 → 3157 | 4975 → 7052 | 838096 → 1187032 | 2973 → 5146 | 7970 → 7758 | +2.66% |
| glyph32:U+0042 | glyph32 | 4470 | ClassCap(Allocated) → ClassCap(Live) | 2650 → 3157 | 4975 → 7052 | 838096 → 1187032 | 2973 → 5146 | 7970 → 7758 | +2.66% |
| glyph16:U+0026 | glyph16 | 9420 | ClassCap(Allocated) → ClassCap(Live) | 4308 → 4796 | 5000 → 7537 | 840168 → 1273048 | 693 → 3285 | 17979 → 17502 | +2.65% |
| glyph32:U+0026 | glyph32 | 9420 | ClassCap(Allocated) → ClassCap(Live) | 4308 → 4796 | 5000 → 7537 | 840168 → 1273048 | 693 → 3285 | 17979 → 17502 | +2.65% |
| glyph16:U+005A | glyph16 | 255 | ClassCap(Allocated) → ClassCap(Live) | 459 → 697 | 3358 → 5513 | 577864 → 943600 | 14056 → 28066 | 393 → 383 | +2.54% |
| glyph16:U+007A | glyph16 | 255 | ClassCap(Allocated) → ClassCap(Live) | 459 → 697 | 3358 → 5513 | 577864 → 943600 | 14056 → 28066 | 393 → 383 | +2.54% |
| glyph32:U+005A | glyph32 | 255 | ClassCap(Allocated) → ClassCap(Live) | 458 → 696 | 3357 → 5512 | 577696 → 943432 | 14056 → 28067 | 393 → 383 | +2.54% |
| glyph32:U+007A | glyph32 | 255 | ClassCap(Allocated) → ClassCap(Live) | 459 → 697 | 3358 → 5513 | 577864 → 943600 | 14056 → 28066 | 393 → 383 | +2.54% |
| glyph16:U+0036 | glyph16 | 8067 | ClassCap(Allocated) → ClassCap(Live) | 3663 → 4576 | 5000 → 8240 | 843248 → 1390256 | 1387 → 4642 | 15249 → 14872 | +2.47% |
| glyph32:U+0036 | glyph32 | 8067 | ClassCap(Allocated) → ClassCap(Live) | 3663 → 4576 | 5000 → 8240 | 843248 → 1390256 | 1387 → 4642 | 15249 → 14872 | +2.47% |
| glyph16:U+0032 | glyph16 | 3891 | ClassCap(Allocated) → ClassCap(Live) | 2533 → 2881 | 4945 → 6465 | 832384 → 1087800 | 3078 → 4761 | 6980 → 6810 | +2.44% |
| glyph32:U+0032 | glyph32 | 3891 | ClassCap(Allocated) → ClassCap(Live) | 2533 → 2881 | 4945 → 6465 | 832384 → 1087800 | 3078 → 4761 | 6980 → 6810 | +2.44% |
| glyph16:U+0077 | glyph16 | 3663 | ClassCap(Allocated) → ClassCap(Live) | 2492 → 2834 | 4937 → 6480 | 831320 → 1090712 | 3133 → 4816 | 6602 → 6443 | +2.41% |
| glyph32:U+0077 | glyph32 | 3663 | ClassCap(Allocated) → ClassCap(Live) | 2492 → 2834 | 4937 → 6480 | 831320 → 1090712 | 3133 → 4816 | 6602 → 6443 | +2.41% |
| glyph16:U+002E | glyph16 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1354 → 2027 | 4681 → 7055 | 798284 → 1198740 | 6411 → 10702 | 2881 → 2813 | +2.36% |
| glyph32:U+002E | glyph32 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1354 → 2027 | 4681 → 7055 | 798284 → 1198740 | 6411 → 10702 | 2881 → 2813 | +2.36% |
| glyph16:U+007E | glyph16 | 5093 | ClassCap(Allocated) → ClassCap(Live) | 2812 → 3527 | 4996 → 7897 | 842184 → 1329608 | 2750 → 5677 | 9370 → 9150 | +2.35% |
| glyph32:U+007E | glyph32 | 5093 | ClassCap(Allocated) → ClassCap(Live) | 2812 → 3527 | 4996 → 7897 | 842184 → 1329608 | 2750 → 5677 | 9370 → 9150 | +2.35% |
| glyph16:U+0075 | glyph16 | 2493 | ClassCap(Allocated) → ClassCap(Live) | 1809 → 2477 | 4898 → 8754 | 835072 → 1484896 | 5123 → 13502 | 4153 → 4056 | +2.34% |
| glyph32:U+0075 | glyph32 | 2493 | ClassCap(Allocated) → ClassCap(Live) | 1809 → 2477 | 4898 → 8754 | 835072 → 1484896 | 5123 → 13502 | 4153 → 4056 | +2.34% |
| glyph16:U+0071 | glyph16 | 5046 | ClassCap(Allocated) → ClassCap(Live) | 2802 → 3472 | 4992 → 7810 | 841400 → 1314824 | 2777 → 5640 | 9172 → 8958 | +2.33% |
| glyph32:U+0071 | glyph32 | 5046 | ClassCap(Allocated) → ClassCap(Live) | 2802 → 3472 | 4992 → 7810 | 841400 → 1314824 | 2777 → 5640 | 9172 → 8958 | +2.33% |
| glyph16:U+0038 | glyph16 | 9774 | ClassCap(Allocated) → ClassCap(Live) | 4420 → 4832 | 5000 → 7225 | 840112 → 1219120 | 581 → 2841 | 18500 → 18085 | +2.24% |
| glyph32:U+0038 | glyph32 | 9774 | ClassCap(Allocated) → ClassCap(Live) | 4420 → 4832 | 5000 → 7225 | 840112 → 1219120 | 581 → 2841 | 18500 → 18085 | +2.24% |
| glyph16:U+0043 | glyph16 | 5129 | ClassCap(Allocated) → ClassCap(Live) | 2830 → 3544 | 4996 → 7876 | 842184 → 1326024 | 2718 → 5633 | 9459 → 9252 | +2.19% |
| glyph32:U+0043 | glyph32 | 5129 | ClassCap(Allocated) → ClassCap(Live) | 2830 → 3544 | 4996 → 7876 | 842184 → 1326024 | 2718 → 5633 | 9459 → 9252 | +2.19% |
| glyph16:U+0062 | glyph16 | 5129 | ClassCap(Allocated) → ClassCap(Live) | 2824 → 3524 | 4995 → 7841 | 841848 → 1320032 | 2720 → 5631 | 9434 → 9228 | +2.18% |
| glyph32:U+0062 | glyph32 | 5129 | ClassCap(Allocated) → ClassCap(Live) | 2824 → 3524 | 4995 → 7841 | 841848 → 1320032 | 2720 → 5631 | 9434 → 9228 | +2.18% |
| glyph16:U+0069 | glyph16 | 2123 | ClassCap(Allocated) → ClassCap(Live) | 1455 → 2225 | 4727 → 7925 | 808752 → 1347248 | 6347 → 12301 | 3510 → 3434 | +2.17% |
| glyph32:U+0069 | glyph32 | 2123 | ClassCap(Allocated) → ClassCap(Live) | 1455 → 2225 | 4727 → 7925 | 808752 → 1347248 | 6347 → 12301 | 3510 → 3434 | +2.17% |
| glyph16:U+0072 | glyph16 | 2456 | ClassCap(Allocated) → ClassCap(Live) | 1804 → 2504 | 4912 → 8977 | 838992 → 1525328 | 5113 → 13808 | 4180 → 4090 | +2.15% |
| glyph32:U+0072 | glyph32 | 2456 | ClassCap(Allocated) → ClassCap(Live) | 1804 → 2504 | 4912 → 8977 | 838992 → 1525328 | 5113 → 13808 | 4180 → 4090 | +2.15% |
| glyph16:U+0074 | glyph16 | 2421 | ClassCap(Allocated) → ClassCap(Live) | 1796 → 2551 | 4905 → 9094 | 837312 → 1542744 | 5091 → 13875 | 4109 → 4021 | +2.14% |
| glyph32:U+0074 | glyph32 | 2421 | ClassCap(Allocated) → ClassCap(Live) | 1796 → 2551 | 4905 → 9094 | 837312 → 1542744 | 5091 → 13875 | 4109 → 4021 | +2.14% |
| glyph16:U+004F | glyph16 | 5945 | ClassCap(Allocated) → ClassCap(Live) | 3006 → 3917 | 5000 → 8206 | 844312 → 1382976 | 2421 → 5650 | 10769 → 10539 | +2.14% |
| glyph32:U+004F | glyph32 | 5945 | ClassCap(Allocated) → ClassCap(Live) | 3006 → 3917 | 5000 → 8206 | 844312 → 1382976 | 2421 → 5650 | 10769 → 10539 | +2.14% |
| glyph16:U+006E | glyph16 | 2264 | ClassCap(Allocated) → ClassCap(Live) | 1631 → 2352 | 4826 → 8378 | 823368 → 1422232 | 5722 → 12898 | 3759 → 3680 | +2.10% |
| glyph32:U+006E | glyph32 | 2264 | ClassCap(Allocated) → ClassCap(Live) | 1631 → 2352 | 4826 → 8378 | 823368 → 1422232 | 5722 → 12898 | 3759 → 3680 | +2.10% |
| glyph16:U+0060 | glyph16 | 1722 | ClassCap(Allocated) → ClassCap(Live) | 1342 → 2016 | 4689 → 7242 | 799680 → 1230432 | 6355 → 11185 | 2827 → 2768 | +2.09% |
| glyph32:U+0060 | glyph32 | 1722 | ClassCap(Allocated) → ClassCap(Live) | 1342 → 2016 | 4689 → 7242 | 799680 → 1230432 | 6355 → 11185 | 2827 → 2768 | +2.09% |
| glyph16:U+006F | glyph16 | 5716 | ClassCap(Allocated) → ClassCap(Live) | 2947 → 3745 | 5000 → 8007 | 843080 → 1348200 | 2515 → 5622 | 10262 → 10052 | +2.05% |
| glyph32:U+006F | glyph32 | 5716 | ClassCap(Allocated) → ClassCap(Live) | 2947 → 3745 | 5000 → 8007 | 843080 → 1348200 | 2515 → 5622 | 10262 → 10052 | +2.05% |
| glyph16:U+0030 | glyph16 | 5402 | ClassCap(Allocated) → ClassCap(Live) | 2903 → 3671 | 5000 → 7934 | 843416 → 1336272 | 2586 → 5574 | 9970 → 9768 | +2.03% |
| glyph32:U+0030 | glyph32 | 5402 | ClassCap(Allocated) → ClassCap(Live) | 2903 → 3671 | 5000 → 7934 | 843416 → 1336272 | 2586 → 5574 | 9970 → 9768 | +2.03% |
| glyph16:U+0066 | glyph16 | 2364 | ClassCap(Allocated) → ClassCap(Live) | 1682 → 2399 | 4880 → 8554 | 833616 → 1452416 | 5469 → 13251 | 3955 → 3875 | +2.02% |
| glyph32:U+0066 | glyph32 | 2364 | ClassCap(Allocated) → ClassCap(Live) | 1682 → 2399 | 4880 → 8554 | 833616 → 1452416 | 5469 → 13251 | 3955 → 3875 | +2.02% |
| glyph16:U+0051 | glyph16 | 6053 | ClassCap(Allocated) → ClassCap(Live) | 3029 → 3941 | 5000 → 8196 | 843304 → 1380176 | 2388 → 5609 | 10749 → 10532 | +2.02% |
| glyph32:U+0051 | glyph32 | 6053 | ClassCap(Allocated) → ClassCap(Live) | 3029 → 3941 | 5000 → 8196 | 843304 → 1380176 | 2388 → 5609 | 10749 → 10532 | +2.02% |
| glyph16:U+0070 | glyph16 | 5406 | ClassCap(Allocated) → ClassCap(Live) | 2901 → 3700 | 4999 → 7992 | 843080 → 1345848 | 2589 → 5630 | 9922 → 9722 | +2.02% |
| glyph32:U+0070 | glyph32 | 5406 | ClassCap(Allocated) → ClassCap(Live) | 2901 → 3700 | 4999 → 7992 | 843080 → 1345848 | 2589 → 5630 | 9922 → 9722 | +2.02% |
| glyph16:U+002C | glyph16 | 1694 | ClassCap(Allocated) → ClassCap(Live) | 1366 → 1963 | 4678 → 6820 | 796768 → 1157408 | 6361 → 10474 | 2779 → 2723 | +2.02% |
| glyph32:U+002C | glyph32 | 1694 | ClassCap(Allocated) → ClassCap(Live) | 1366 → 1963 | 4678 → 6820 | 796768 → 1157408 | 6361 → 10474 | 2779 → 2723 | +2.02% |
| glyph16:U+0050 | glyph16 | 2409 | ClassCap(Allocated) → ClassCap(Live) | 1714 → 2453 | 4880 → 8681 | 832496 → 1472968 | 5418 → 13199 | 3990 → 3910 | +2.01% |
| glyph32:U+0050 | glyph32 | 2409 | ClassCap(Allocated) → ClassCap(Live) | 1714 → 2453 | 4880 → 8681 | 832496 → 1472968 | 5418 → 13199 | 3990 → 3910 | +2.01% |
| glyph16:U+0021 | glyph16 | 1927 | ClassCap(Allocated) → ClassCap(Live) | 1352 → 2118 | 4703 → 7369 | 802536 → 1251320 | 6357 → 11245 | 3125 → 3064 | +1.95% |
| glyph32:U+0021 | glyph32 | 1927 | ClassCap(Allocated) → ClassCap(Live) | 1352 → 2118 | 4703 → 7369 | 802536 → 1251320 | 6357 → 11245 | 3125 → 3064 | +1.95% |
| glyph16:U+0064 | glyph16 | 5864 | ClassCap(Allocated) → ClassCap(Live) | 3022 → 3940 | 5000 → 8182 | 843584 → 1378160 | 2388 → 5577 | 10763 → 10560 | +1.89% |
| glyph32:U+0064 | glyph32 | 5864 | ClassCap(Allocated) → ClassCap(Live) | 3022 → 3940 | 5000 → 8182 | 843584 → 1378160 | 2388 → 5577 | 10763 → 10560 | +1.89% |
| glyph16:U+0061 | glyph16 | 5743 | ClassCap(Allocated) → ClassCap(Live) | 3007 → 3907 | 5000 → 8140 | 843304 → 1370824 | 2411 → 5560 | 10612 → 10414 | +1.87% |
| glyph32:U+0061 | glyph32 | 5743 | ClassCap(Allocated) → ClassCap(Live) | 3007 → 3907 | 5000 → 8140 | 843304 → 1370824 | 2411 → 5560 | 10612 → 10414 | +1.87% |
| glyph16:U+0025 | glyph16 | 7327 | ClassCap(Allocated) → ClassCap(Live) | 3186 → 4232 | 5000 → 8395 | 844760 → 1414784 | 1990 → 5480 | 12640 → 12430 | +1.66% |
| glyph32:U+0025 | glyph32 | 7327 | ClassCap(Allocated) → ClassCap(Live) | 3186 → 4232 | 5000 → 8395 | 844760 → 1414784 | 1990 → 5480 | 12640 → 12430 | +1.66% |
| glyph16:U+0028 | glyph16 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1379 → 2067 | 4719 → 7132 | 804048 → 1210328 | 6310 → 10875 | 3039 → 2989 | +1.65% |
| glyph16:U+0029 | glyph16 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1378 → 2078 | 4718 → 7162 | 804328 → 1216040 | 6310 → 10894 | 3039 → 2989 | +1.65% |
| glyph32:U+0028 | glyph32 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1379 → 2067 | 4719 → 7132 | 804048 → 1210328 | 6310 → 10875 | 3039 → 2989 | +1.65% |
| glyph32:U+0029 | glyph32 | 1855 | ClassCap(Allocated) → ClassCap(Live) | 1378 → 2078 | 4718 → 7162 | 804328 → 1216040 | 6310 → 10894 | 3039 → 2989 | +1.65% |
| glyph16:U+0073 | glyph16 | 6929 | ClassCap(Allocated) → ClassCap(Live) | 3201 → 4256 | 5000 → 8360 | 844928 → 1409016 | 1968 → 5365 | 12680 → 12477 | +1.60% |
| glyph32:U+0073 | glyph32 | 6929 | ClassCap(Allocated) → ClassCap(Live) | 3201 → 4256 | 5000 → 8360 | 844928 → 1409016 | 1968 → 5365 | 12680 → 12477 | +1.60% |
| glyph16:U+003F | glyph16 | 6716 | ClassCap(Allocated) → ClassCap(Live) | 3176 → 4176 | 5000 → 8320 | 844648 → 1402240 | 2009 → 5431 | 12176 → 11985 | +1.57% |
| glyph32:U+003F | glyph32 | 6716 | ClassCap(Allocated) → ClassCap(Live) | 3176 → 4176 | 5000 → 8320 | 844648 → 1402240 | 2009 → 5431 | 12176 → 11985 | +1.57% |
| glyph16:U+0033 | glyph16 | 6993 | ClassCap(Allocated) → ClassCap(Live) | 3209 → 4291 | 5000 → 8381 | 845096 → 1412936 | 1948 → 5347 | 12797 → 12605 | +1.50% |
| glyph32:U+0033 | glyph32 | 6993 | ClassCap(Allocated) → ClassCap(Live) | 3209 → 4291 | 5000 → 8381 | 845096 → 1412936 | 1948 → 5347 | 12797 → 12605 | +1.50% |
| glyph16:U+0067 | glyph16 | 7009 | ClassCap(Allocated) → ClassCap(Live) | 3208 → 4282 | 5000 → 8366 | 844088 → 1409408 | 1951 → 5329 | 12799 → 12607 | +1.50% |
| glyph32:U+0067 | glyph32 | 7009 | ClassCap(Allocated) → ClassCap(Live) | 3208 → 4282 | 5000 → 8366 | 844088 → 1409408 | 1951 → 5329 | 12799 → 12607 | +1.50% |
| shader:cosine_palette | shader | 40 | ClassCap(Allocated) → ClassCap(Live) | 474 → 1237 | 1767 → 4545 | 298760 → 770616 | 2614 → 7384 | 296 → 292 | +1.35% |
| glyph16:U+0057 | glyph16 | 7232 | ClassCap(Allocated) → ClassCap(Live) | 3271 → 4426 | 5000 → 8424 | 846104 → 1421448 | 1842 → 5158 | 13499 → 13367 | +0.98% |
| glyph32:U+0057 | glyph32 | 7232 | ClassCap(Allocated) → ClassCap(Live) | 3271 → 4426 | 5000 → 8424 | 846104 → 1421448 | 1842 → 5158 | 13499 → 13367 | +0.98% |
| glyph16:U+0024 | glyph16 | 7185 | ClassCap(Allocated) → ClassCap(Live) | 3246 → 4355 | 5000 → 8398 | 844200 → 1415120 | 1886 → 5238 | 13131 → 13008 | +0.94% |
| glyph32:U+0024 | glyph32 | 7185 | ClassCap(Allocated) → ClassCap(Live) | 3246 → 4355 | 5000 → 8398 | 844200 → 1415120 | 1886 → 5238 | 13131 → 13008 | +0.94% |
| glyph16:U+0053 | glyph16 | 7616 | ClassCap(Allocated) → ClassCap(Live) | 3414 → 4444 | 5000 → 8316 | 844480 → 1402632 | 1682 → 4989 | 14066 → 13956 | +0.78% |
| glyph32:U+0053 | glyph32 | 7616 | ClassCap(Allocated) → ClassCap(Live) | 3414 → 4444 | 5000 → 8316 | 844480 → 1402632 | 1682 → 4989 | 14066 → 13956 | +0.78% |
| glyph16:U+004D | glyph16 | 2345 | ClassCap(Allocated) → ClassCap(Live) | 1703 → 2612 | 4914 → 9209 | 843528 → 1568392 | 5331 → 13382 | 4133 → 4121 | +0.29% |
| glyph32:U+004D | glyph32 | 2345 | ClassCap(Allocated) → ClassCap(Live) | 1703 → 2612 | 4914 → 9209 | 843528 → 1568392 | 5331 → 13382 | 4133 → 4121 | +0.29% |
| synth_d11_s39 | synthetic | 475 | ClassCap(Allocated) → ClassCap(Live) | 813 → 1134 | 3509 → 7330 | 596344 → 1264648 | 11218 → 24916 | 6791 → 6778 | +0.19% |
| glyph16:U+0079 | glyph16 | 3530 | ClassCap(Allocated) → ClassCap(Live) | 2436 → 2696 | 4915 → 6172 | 827232 → 1038408 | 3229 → 4610 | 6403 → 6393 | +0.16% |
| glyph32:U+0079 | glyph32 | 3530 | ClassCap(Allocated) → ClassCap(Live) | 2436 → 2696 | 4915 → 6172 | 827232 → 1038408 | 3229 → 4610 | 6403 → 6393 | +0.16% |
| glyph16:U+007B | glyph16 | 4598 | ClassCap(Allocated) → ClassCap(Live) | 2694 → 3259 | 4984 → 7306 | 840000 → 1230096 | 2918 → 5301 | 8205 → 8195 | +0.12% |
| glyph32:U+007B | glyph32 | 4598 | ClassCap(Allocated) → ClassCap(Live) | 2694 → 3259 | 4984 → 7306 | 840000 → 1230096 | 2918 → 5301 | 8205 → 8195 | +0.12% |
| glyph16:U+007D | glyph16 | 4598 | ClassCap(Allocated) → ClassCap(Live) | 2688 → 3231 | 4983 → 7220 | 839440 → 1215312 | 2918 → 5239 | 8215 → 8205 | +0.12% |
| glyph32:U+007D | glyph32 | 4598 | ClassCap(Allocated) → ClassCap(Live) | 2688 → 3231 | 4983 → 7220 | 839440 → 1215312 | 2918 → 5239 | 8215 → 8205 | +0.12% |
| glyph16:U+0063 | glyph16 | 4671 | ClassCap(Allocated) → ClassCap(Live) | 2700 → 3265 | 4984 → 7298 | 839720 → 1228472 | 2923 → 5288 | 8419 → 8409 | +0.12% |
| glyph32:U+0063 | glyph32 | 4671 | ClassCap(Allocated) → ClassCap(Live) | 2700 → 3265 | 4984 → 7298 | 839720 → 1228472 | 2923 → 5288 | 8419 → 8409 | +0.12% |
| glyph16:U+0065 | glyph16 | 4872 | ClassCap(Allocated) → ClassCap(Live) | 2751 → 3387 | 4989 → 7585 | 840672 → 1276800 | 2865 → 5455 | 8766 → 8756 | +0.11% |
| glyph32:U+0065 | glyph32 | 4872 | ClassCap(Allocated) → ClassCap(Live) | 2751 → 3387 | 4989 → 7585 | 840672 → 1276800 | 2865 → 5455 | 8766 → 8756 | +0.11% |
| glyph16:U+0047 | glyph16 | 4563 | ClassCap(Allocated) → ClassCap(Live) | 2695 → 3250 | 4983 → 7266 | 839216 → 1222760 | 2915 → 5258 | 8262 → 8258 | +0.05% |
| glyph32:U+0047 | glyph32 | 4563 | ClassCap(Allocated) → ClassCap(Live) | 2695 → 3250 | 4983 → 7266 | 839216 → 1222760 | 2915 → 5258 | 8262 → 8258 | +0.05% |
| synth_d11_s3 | synthetic | 835 | ClassCap(Allocated) → ClassCap(Live) | 1217 → 2195 | 4664 → 8714 | 794024 → 1479968 | 7807 → 14017 | 11719 → 11718 | +0.01% |
| cellgrid:120x40_d2 | cellgrid | 623 | ClassCap(Allocated) → ClassCap(Live) | 1261 → 1763 | 4583 → 8034 | 778512 → 1361360 | 6949 → 11605 | 432 → 432 | +0.00% |
| cellgrid:80x24_d1 | cellgrid | 623 | ClassCap(Allocated) → ClassCap(Live) | 1260 → 1760 | 4583 → 8030 | 778512 → 1360688 | 6946 → 11602 | 427 → 427 | +0.00% |
| cellgrid:80x24_d2 | cellgrid | 623 | ClassCap(Allocated) → ClassCap(Live) | 1261 → 1763 | 4583 → 8034 | 778512 → 1361360 | 6949 → 11605 | 432 → 432 | +0.00% |
| glyph16:U+0020 | glyph16 | 1 | Quiesced → Quiesced | 1 → 1 | 1 → 1 | 168 → 168 | 0 → 0 | 0 → 0 | +0.00% |
| glyph16:U+0023 | glyph16 | 615 | ClassCap(Allocated) → ClassCap(Live) | 826 → 922 | 4309 → 5541 | 744352 → 958664 | 9842 → 13549 | 1014 → 1014 | +0.00% |
| glyph16:U+002A | glyph16 | 559 | ClassCap(Allocated) → ClassCap(Live) | 1002 → 1104 | 4480 → 5729 | 772128 → 990192 | 9666 → 13789 | 1143 → 1143 | +0.00% |
| glyph16:U+002B | glyph16 | 247 | ClassCap(Allocated) → Quiesced | 244 → 254 | 2055 → 2346 | 355824 → 404768 | 11436 → 60676 | 319 → 319 | +0.00% |
| glyph16:U+002D | glyph16 | 119 | Quiesced → Quiesced | 117 → 117 | 1035 → 1035 | 181160 → 181160 | 15080 → 15080 | 133 → 133 | +0.00% |
| glyph16:U+0035 | glyph16 | 4546 | ClassCap(Allocated) → ClassCap(Live) | 2683 → 3212 | 4981 → 7177 | 838824 → 1207752 | 2926 → 5204 | 8145 → 8145 | +0.00% |
| glyph16:U+003A | glyph16 | 3655 | ClassCap(Allocated) → ClassCap(Live) | 2312 → 2535 | 4854 → 5834 | 816424 → 981064 | 3374 → 4429 | 6228 → 6228 | +0.00% |
| glyph16:U+003D | glyph16 | 183 | Quiesced → Quiesced | 172 → 172 | 1514 → 1514 | 263200 → 263200 | 21429 → 21429 | 231 → 231 | +0.00% |
| glyph16:U+0040 | glyph16 | 12056 | ClassCap(Allocated) → ClassCap(Live) | 5226 → 5226 | 5226 → 5226 | 877968 → 877968 | 0 → 0 | 23527 → 23527 | +0.00% |
| glyph16:U+0045 | glyph16 | 247 | ClassCap(Allocated) → ClassCap(Live) | 370 → 556 | 2780 → 4623 | 477848 → 790664 | 12806 → 24259 | 335 → 335 | +0.00% |
| glyph16:U+0046 | glyph16 | 215 | ClassCap(Allocated) → ClassCap(Live) | 331 → 508 | 2535 → 4252 | 435848 → 727552 | 11996 → 22382 | 285 → 285 | +0.00% |
| glyph16:U+0048 | glyph16 | 247 | ClassCap(Allocated) → ClassCap(Live) | 369 → 562 | 2793 → 4724 | 479920 → 807576 | 12700 → 24358 | 314 → 314 | +0.00% |
| glyph16:U+004C | glyph16 | 151 | ClassCap(Allocated) → ClassCap(Live) | 422 → 642 | 3361 → 5681 | 575904 → 968632 | 17318 → 32765 | 185 → 185 | +0.00% |
| glyph16:U+0054 | glyph16 | 183 | ClassCap(Allocated) → ClassCap(Live) | 468 → 684 | 3521 → 5906 | 602672 → 1007440 | 18538 → 35113 | 227 → 227 | +0.00% |
| glyph16:U+005B | glyph16 | 183 | Quiesced → Quiesced | 175 → 175 | 1519 → 1519 | 264040 → 264040 | 21494 → 21494 | 234 → 234 | +0.00% |
| glyph16:U+005D | glyph16 | 183 | Quiesced → Quiesced | 177 → 177 | 1517 → 1517 | 263704 → 263704 | 21364 → 21364 | 244 → 244 | +0.00% |
| glyph16:U+005F | glyph16 | 119 | ClassCap(Allocated) → ClassCap(Live) | 321 → 566 | 2994 → 4836 | 512960 → 825888 | 23449 → 38923 | 129 → 129 | +0.00% |
| glyph16:U+006A | glyph16 | 4293 | ClassCap(Allocated) → ClassCap(Live) | 2597 → 3013 | 4971 → 6782 | 836808 → 1141112 | 3014 → 4994 | 7523 → 7523 | +0.00% |
| glyph16:U+007C | glyph16 | 119 | Quiesced → Quiesced | 112 → 112 | 1044 → 1044 | 182896 → 182896 | 15325 → 15325 | 133 → 133 | +0.00% |
| glyph32:U+0020 | glyph32 | 1 | Quiesced → Quiesced | 1 → 1 | 1 → 1 | 168 → 168 | 0 → 0 | 0 → 0 | +0.00% |
| glyph32:U+0023 | glyph32 | 615 | ClassCap(Allocated) → ClassCap(Live) | 826 → 922 | 4309 → 5541 | 744352 → 958664 | 9842 → 13549 | 1014 → 1014 | +0.00% |
| glyph32:U+002A | glyph32 | 559 | ClassCap(Allocated) → ClassCap(Live) | 1002 → 1104 | 4480 → 5729 | 772128 → 990192 | 9666 → 13789 | 1143 → 1143 | +0.00% |
| glyph32:U+002B | glyph32 | 247 | ClassCap(Allocated) → Quiesced | 244 → 254 | 2055 → 2346 | 355824 → 404768 | 11436 → 60676 | 319 → 319 | +0.00% |
| glyph32:U+002D | glyph32 | 119 | Quiesced → Quiesced | 116 → 116 | 1034 → 1034 | 180992 → 180992 | 15080 → 15080 | 133 → 133 | +0.00% |
| glyph32:U+0035 | glyph32 | 4546 | ClassCap(Allocated) → ClassCap(Live) | 2683 → 3212 | 4981 → 7177 | 838824 → 1207752 | 2926 → 5204 | 8145 → 8145 | +0.00% |
| glyph32:U+003A | glyph32 | 3655 | ClassCap(Allocated) → ClassCap(Live) | 2312 → 2535 | 4854 → 5834 | 816424 → 981064 | 3374 → 4429 | 6228 → 6228 | +0.00% |
| glyph32:U+003D | glyph32 | 183 | Quiesced → Quiesced | 172 → 172 | 1514 → 1514 | 263200 → 263200 | 21429 → 21429 | 231 → 231 | +0.00% |
| glyph32:U+0040 | glyph32 | 12056 | ClassCap(Allocated) → ClassCap(Live) | 5226 → 5226 | 5226 → 5226 | 877968 → 877968 | 0 → 0 | 23527 → 23527 | +0.00% |
| glyph32:U+0045 | glyph32 | 247 | ClassCap(Allocated) → ClassCap(Live) | 369 → 555 | 2779 → 4622 | 477680 → 790496 | 12807 → 24259 | 335 → 335 | +0.00% |
| glyph32:U+0046 | glyph32 | 215 | ClassCap(Allocated) → ClassCap(Live) | 330 → 507 | 2534 → 4251 | 435680 → 727384 | 11996 → 22382 | 285 → 285 | +0.00% |
| glyph32:U+0048 | glyph32 | 247 | ClassCap(Allocated) → ClassCap(Live) | 368 → 561 | 2792 → 4723 | 479752 → 807408 | 12701 → 24358 | 314 → 314 | +0.00% |
| glyph32:U+004C | glyph32 | 151 | ClassCap(Allocated) → ClassCap(Live) | 421 → 641 | 3360 → 5680 | 575736 → 968464 | 17319 → 32765 | 185 → 185 | +0.00% |
| glyph32:U+0054 | glyph32 | 183 | ClassCap(Allocated) → ClassCap(Live) | 467 → 683 | 3520 → 5905 | 602504 → 1007272 | 18538 → 35115 | 227 → 227 | +0.00% |
| glyph32:U+005B | glyph32 | 183 | Quiesced → Quiesced | 175 → 175 | 1519 → 1519 | 264040 → 264040 | 21494 → 21494 | 234 → 234 | +0.00% |
| glyph32:U+005D | glyph32 | 183 | Quiesced → Quiesced | 177 → 177 | 1517 → 1517 | 263704 → 263704 | 21364 → 21364 | 244 → 244 | +0.00% |
| glyph32:U+005F | glyph32 | 119 | ClassCap(Allocated) → ClassCap(Live) | 321 → 566 | 2994 → 4836 | 512960 → 825888 | 23449 → 38923 | 129 → 129 | +0.00% |
| glyph32:U+006A | glyph32 | 4293 | ClassCap(Allocated) → ClassCap(Live) | 2597 → 3013 | 4971 → 6782 | 836808 → 1141112 | 3014 → 4994 | 7523 → 7523 | +0.00% |
| glyph32:U+007C | glyph32 | 119 | Quiesced → Quiesced | 111 → 111 | 1043 → 1043 | 182728 → 182728 | 15325 → 15325 | 133 → 133 | +0.00% |
| shader:gyroid_slice | shader | 44 | Quiesced → Quiesced | 104 → 104 | 779 → 779 | 132944 → 132944 | 8652 → 8652 | 938 → 938 | +0.00% |
| shader:kaleidoscope_fold | shader | 46 | Quiesced → Quiesced | 57 → 57 | 131 → 131 | 22008 → 22008 | 601 → 601 | 554 → 554 | +0.00% |
| shader:plasma | shader | 41 | ClassCap(Allocated) → ClassCap(Live) | 336 → 514 | 1822 → 2752 | 308336 → 465024 | 3310 → 5324 | 363 → 363 | +0.00% |
| shader:smoothstep_vignette | shader | 64 | Quiesced → Quiesced | 82 → 82 | 283 → 283 | 48216 → 48216 | 1596 → 1596 | 161 → 161 | +0.00% |
| shader:star_sdf | shader | 66 | ClassCap(Allocated) → ClassCap(Live) | 632 → 822 | 3828 → 5145 | 657720 → 880768 | 7997 → 9641 | 169 → 169 | +0.00% |
| shader:torus_slice | shader | 42 | ClassCap(Allocated) → ClassCap(Live) | 166 → 205 | 1291 → 1988 | 223104 → 342888 | 4697 → 8529 | 141 → 141 | +0.00% |
| synth_d3_s0 | synthetic | 19 | Quiesced → Quiesced | 17 → 17 | 37 → 37 | 6328 → 6328 | 99 → 99 | 163 → 163 | +0.00% |
| synth_d3_s1 | synthetic | 30 | Quiesced → Quiesced | 18 → 18 | 35 → 35 | 5992 → 5992 | 63 → 63 | 361 → 361 | +0.00% |
| synth_d3_s2 | synthetic | 14 | Quiesced → Quiesced | 12 → 12 | 18 → 18 | 3080 → 3080 | 19 → 19 | 318 → 318 | +0.00% |
| synth_d3_s3 | synthetic | 30 | Quiesced → Quiesced | 22 → 22 | 39 → 39 | 6720 → 6720 | 48 → 48 | 325 → 325 | +0.00% |
| synth_d3_s4 | synthetic | 24 | Quiesced → Quiesced | 31 → 31 | 64 → 64 | 10864 → 10864 | 194 → 194 | 188 → 188 | +0.00% |
| synth_d3_s5 | synthetic | 18 | Quiesced → Quiesced | 18 → 18 | 45 → 45 | 7616 → 7616 | 124 → 124 | 40 → 40 | +0.00% |
| synth_d3_s6 | synthetic | 42 | Quiesced → Quiesced | 78 → 78 | 461 → 461 | 78736 → 78736 | 4622 → 4622 | 240 → 240 | +0.00% |
| synth_d3_s7 | synthetic | 6 | Quiesced → Quiesced | 5 → 5 | 7 → 7 | 1232 → 1232 | 6 → 6 | 219 → 219 | +0.00% |
| synth_d3_s8 | synthetic | 9 | Quiesced → Quiesced | 8 → 8 | 12 → 12 | 2072 → 2072 | 11 → 11 | 98 → 98 | +0.00% |
| synth_d3_s9 | synthetic | 14 | Quiesced → Quiesced | 12 → 12 | 19 → 19 | 3248 → 3248 | 20 → 20 | 33 → 33 | +0.00% |
| synth_d3_s10 | synthetic | 20 | Quiesced → Quiesced | 20 → 20 | 43 → 43 | 7392 → 7392 | 130 → 130 | 90 → 90 | +0.00% |
| synth_d3_s11 | synthetic | 16 | Quiesced → Quiesced | 22 → 22 | 67 → 67 | 11256 → 11256 | 346 → 346 | 232 → 232 | +0.00% |
| synth_d3_s12 | synthetic | 13 | Quiesced → Quiesced | 10 → 10 | 17 → 17 | 2912 → 2912 | 21 → 21 | 149 → 149 | +0.00% |
| synth_d3_s13 | synthetic | 16 | Quiesced → Quiesced | 12 → 12 | 21 → 21 | 3640 → 3640 | 44 → 44 | 280 → 280 | +0.00% |
| synth_d3_s14 | synthetic | 26 | Quiesced → Quiesced | 19 → 19 | 38 → 38 | 6496 → 6496 | 119 → 119 | 235 → 235 | +0.00% |
| synth_d3_s15 | synthetic | 13 | Quiesced → Quiesced | 10 → 10 | 17 → 17 | 2968 → 2968 | 20 → 20 | 208 → 208 | +0.00% |
| synth_d3_s16 | synthetic | 21 | Quiesced → Quiesced | 19 → 19 | 49 → 49 | 8456 → 8456 | 139 → 139 | 153 → 153 | +0.00% |
| synth_d3_s17 | synthetic | 37 | Quiesced → Quiesced | 26 → 26 | 44 → 44 | 7560 → 7560 | 51 → 51 | 400 → 400 | +0.00% |
| synth_d3_s18 | synthetic | 16 | Quiesced → Quiesced | 12 → 12 | 19 → 19 | 3304 → 3304 | 20 → 20 | 343 → 343 | +0.00% |
| synth_d3_s19 | synthetic | 26 | Quiesced → Quiesced | 18 → 18 | 32 → 32 | 5376 → 5376 | 61 → 61 | 614 → 614 | +0.00% |
| synth_d3_s20 | synthetic | 9 | Quiesced → Quiesced | 9 → 9 | 14 → 14 | 2352 → 2352 | 11 → 11 | 176 → 176 | +0.00% |
| synth_d3_s21 | synthetic | 17 | Quiesced → Quiesced | 16 → 16 | 23 → 23 | 3864 → 3864 | 21 → 21 | 302 → 302 | +0.00% |
| synth_d3_s22 | synthetic | 8 | Quiesced → Quiesced | 6 → 6 | 10 → 10 | 1736 → 1736 | 11 → 11 | 81 → 81 | +0.00% |
| synth_d3_s23 | synthetic | 35 | ClassCap(Allocated) → ClassCap(Live) | 333 → 466 | 1704 → 3734 | 289464 → 636552 | 2948 → 9093 | 132 → 132 | +0.00% |
| synth_d3_s24 | synthetic | 17 | Quiesced → Quiesced | 17 → 17 | 34 → 34 | 5824 → 5824 | 95 → 95 | 126 → 126 | +0.00% |
| synth_d3_s25 | synthetic | 7 | Quiesced → Quiesced | 5 → 5 | 8 → 8 | 1400 → 1400 | 8 → 8 | 237 → 237 | +0.00% |
| synth_d3_s26 | synthetic | 10 | Quiesced → Quiesced | 7 → 7 | 13 → 13 | 2296 → 2296 | 17 → 17 | 83 → 83 | +0.00% |
| synth_d3_s27 | synthetic | 35 | Quiesced → Quiesced | 29 → 29 | 65 → 65 | 11032 → 11032 | 175 → 175 | 455 → 455 | +0.00% |
| synth_d3_s28 | synthetic | 29 | Quiesced → Quiesced | 29 → 29 | 93 → 93 | 16016 → 16016 | 423 → 423 | 280 → 280 | +0.00% |
| synth_d3_s29 | synthetic | 26 | Quiesced → Quiesced | 23 → 23 | 60 → 60 | 10136 → 10136 | 202 → 202 | 360 → 360 | +0.00% |
| synth_d3_s30 | synthetic | 12 | Quiesced → Quiesced | 11 → 11 | 16 → 16 | 2744 → 2744 | 14 → 14 | 104 → 104 | +0.00% |
| synth_d3_s31 | synthetic | 30 | Quiesced → Quiesced | 19 → 19 | 47 → 47 | 8120 → 8120 | 263 → 263 | 122 → 122 | +0.00% |
| synth_d3_s32 | synthetic | 39 | Quiesced → Quiesced | 33 → 33 | 73 → 73 | 12488 → 12488 | 250 → 250 | 324 → 324 | +0.00% |
| synth_d3_s33 | synthetic | 21 | Quiesced → Quiesced | 27 → 27 | 88 → 88 | 14840 → 14840 | 441 → 441 | 33 → 33 | +0.00% |
| synth_d3_s34 | synthetic | 9 | Quiesced → Quiesced | 8 → 8 | 12 → 12 | 2072 → 2072 | 11 → 11 | 25 → 25 | +0.00% |
| synth_d3_s35 | synthetic | 35 | Quiesced → Quiesced | 26 → 26 | 47 → 47 | 7952 → 7952 | 125 → 125 | 430 → 430 | +0.00% |
| synth_d3_s36 | synthetic | 28 | Quiesced → Quiesced | 20 → 20 | 37 → 37 | 6496 → 6496 | 80 → 80 | 446 → 446 | +0.00% |
| synth_d3_s37 | synthetic | 12 | Quiesced → Quiesced | 9 → 9 | 14 → 14 | 2408 → 2408 | 13 → 13 | 172 → 172 | +0.00% |
| synth_d3_s38 | synthetic | 14 | Quiesced → Quiesced | 10 → 10 | 17 → 17 | 2912 → 2912 | 18 → 18 | 99 → 99 | +0.00% |
| synth_d3_s39 | synthetic | 15 | Quiesced → Quiesced | 13 → 13 | 23 → 23 | 3920 → 3920 | 24 → 24 | 43 → 43 | +0.00% |
| synth_d5_s0 | synthetic | 22 | Quiesced → Quiesced | 21 → 21 | 34 → 34 | 5768 → 5768 | 76 → 76 | 603 → 603 | +0.00% |
| synth_d5_s1 | synthetic | 58 | Quiesced → Quiesced | 105 → 105 | 705 → 705 | 120288 → 120288 | 5822 → 5822 | 441 → 441 | +0.00% |
| synth_d5_s2 | synthetic | 50 | Quiesced → Quiesced | 42 → 42 | 96 → 96 | 16408 → 16408 | 380 → 380 | 532 → 532 | +0.00% |
| synth_d5_s3 | synthetic | 84 | Quiesced → Quiesced | 177 → 177 | 1123 → 1123 | 192696 → 192696 | 11408 → 11408 | 959 → 959 | +0.00% |
| synth_d5_s4 | synthetic | 63 | Quiesced → Quiesced | 74 → 74 | 314 → 314 | 53704 → 53704 | 2100 → 2100 | 388 → 388 | +0.00% |
| synth_d5_s5 | synthetic | 6 | Quiesced → Quiesced | 6 → 6 | 7 → 7 | 1176 → 1176 | 3 → 3 | 242 → 242 | +0.00% |
| synth_d5_s6 | synthetic | 55 | Quiesced → Quiesced | 40 → 40 | 85 → 85 | 14448 → 14448 | 268 → 268 | 394 → 394 | +0.00% |
| synth_d5_s7 | synthetic | 51 | Quiesced → Quiesced | 74 → 74 | 356 → 356 | 60648 → 60648 | 3431 → 3431 | 906 → 906 | +0.00% |
| synth_d5_s8 | synthetic | 71 | Quiesced → Quiesced | 81 → 81 | 293 → 293 | 49784 → 49784 | 2415 → 2415 | 652 → 652 | +0.00% |
| synth_d5_s9 | synthetic | 52 | Quiesced → Quiesced | 32 → 32 | 62 → 62 | 10752 → 10752 | 122 → 122 | 631 → 631 | +0.00% |
| synth_d5_s10 | synthetic | 40 | Quiesced → Quiesced | 46 → 46 | 170 → 170 | 28728 → 28728 | 934 → 934 | 336 → 336 | +0.00% |
| synth_d5_s11 | synthetic | 81 | ClassCap(Allocated) → Quiesced | 363 → 338 | 3656 → 4257 | 625744 → 730072 | 16867 → 71711 | 851 → 851 | +0.00% |
| synth_d5_s12 | synthetic | 50 | Quiesced → Quiesced | 78 → 78 | 259 → 259 | 43960 → 43960 | 1449 → 1449 | 820 → 820 | +0.00% |
| synth_d5_s13 | synthetic | 21 | Quiesced → Quiesced | 9 → 9 | 28 → 28 | 4984 → 4984 | 94 → 94 | 133 → 133 | +0.00% |
| synth_d5_s14 | synthetic | 34 | Quiesced → Quiesced | 25 → 25 | 42 → 42 | 7280 → 7280 | 49 → 49 | 442 → 442 | +0.00% |
| synth_d5_s15 | synthetic | 114 | Quiesced → Quiesced | 99 → 99 | 250 → 250 | 42336 → 42336 | 1160 → 1160 | 1740 → 1740 | +0.00% |
| synth_d5_s16 | synthetic | 25 | Quiesced → Quiesced | 19 → 19 | 33 → 33 | 5712 → 5712 | 37 → 37 | 391 → 391 | +0.00% |
| synth_d5_s17 | synthetic | 43 | Quiesced → Quiesced | 32 → 32 | 60 → 60 | 10360 → 10360 | 160 → 160 | 443 → 443 | +0.00% |
| synth_d5_s18 | synthetic | 49 | Quiesced → Quiesced | 40 → 40 | 66 → 66 | 11200 → 11200 | 71 → 71 | 386 → 386 | +0.00% |
| synth_d5_s19 | synthetic | 22 | Quiesced → Quiesced | 20 → 20 | 39 → 39 | 6608 → 6608 | 92 → 92 | 211 → 211 | +0.00% |
| synth_d5_s20 | synthetic | 48 | Quiesced → Quiesced | 50 → 50 | 133 → 133 | 22680 → 22680 | 598 → 598 | 394 → 394 | +0.00% |
| synth_d5_s21 | synthetic | 42 | Quiesced → Quiesced | 31 → 31 | 50 → 50 | 8568 → 8568 | 51 → 51 | 679 → 679 | +0.00% |
| synth_d5_s22 | synthetic | 56 | Quiesced → Quiesced | 47 → 47 | 123 → 123 | 20832 → 20832 | 573 → 573 | 915 → 915 | +0.00% |
| synth_d5_s23 | synthetic | 63 | Quiesced → Quiesced | 44 → 44 | 82 → 82 | 14112 → 14112 | 169 → 169 | 759 → 759 | +0.00% |
| synth_d5_s24 | synthetic | 49 | Quiesced → Quiesced | 61 → 61 | 218 → 218 | 37240 → 37240 | 1775 → 1775 | 823 → 823 | +0.00% |
| synth_d5_s25 | synthetic | 22 | Quiesced → Quiesced | 17 → 17 | 30 → 30 | 5152 → 5152 | 36 → 36 | 237 → 237 | +0.00% |
| synth_d5_s26 | synthetic | 44 | Quiesced → Quiesced | 40 → 40 | 93 → 93 | 15848 → 15848 | 348 → 348 | 793 → 793 | +0.00% |
| synth_d5_s27 | synthetic | 27 | Quiesced → Quiesced | 23 → 23 | 48 → 48 | 8232 → 8232 | 163 → 163 | 356 → 356 | +0.00% |
| synth_d5_s28 | synthetic | 61 | Quiesced → Quiesced | 48 → 48 | 93 → 93 | 15848 → 15848 | 203 → 203 | 679 → 679 | +0.00% |
| synth_d5_s29 | synthetic | 69 | Quiesced → Quiesced | 59 → 59 | 115 → 115 | 19488 → 19488 | 400 → 400 | 1140 → 1140 | +0.00% |
| synth_d5_s30 | synthetic | 34 | Quiesced → Quiesced | 27 → 27 | 51 → 51 | 8736 → 8736 | 117 → 117 | 471 → 471 | +0.00% |
| synth_d5_s31 | synthetic | 56 | Quiesced → Quiesced | 59 → 59 | 171 → 171 | 28896 → 28896 | 916 → 916 | 588 → 588 | +0.00% |
| synth_d5_s32 | synthetic | 109 | Quiesced → Quiesced | 97 → 97 | 307 → 307 | 52080 → 52080 | 2311 → 2311 | 1153 → 1153 | +0.00% |
| synth_d5_s33 | synthetic | 42 | Quiesced → Quiesced | 36 → 36 | 69 → 69 | 11816 → 11816 | 152 → 152 | 420 → 420 | +0.00% |
| synth_d5_s34 | synthetic | 64 | Quiesced → Quiesced | 67 → 67 | 183 → 183 | 30968 → 30968 | 765 → 765 | 685 → 685 | +0.00% |
| synth_d5_s35 | synthetic | 74 | Quiesced → Quiesced | 68 → 68 | 172 → 172 | 29120 → 29120 | 986 → 986 | 889 → 889 | +0.00% |
| synth_d5_s36 | synthetic | 21 | Quiesced → Quiesced | 16 → 16 | 26 → 26 | 4480 → 4480 | 29 → 29 | 445 → 445 | +0.00% |
| synth_d5_s37 | synthetic | 74 | Quiesced → Quiesced | 99 → 99 | 337 → 337 | 57344 → 57344 | 2055 → 2055 | 1147 → 1147 | +0.00% |
| synth_d5_s38 | synthetic | 38 | Quiesced → Quiesced | 29 → 29 | 43 → 43 | 7336 → 7336 | 38 → 38 | 824 → 824 | +0.00% |
| synth_d5_s39 | synthetic | 28 | Quiesced → Quiesced | 21 → 21 | 34 → 34 | 5880 → 5880 | 35 → 35 | 591 → 591 | +0.00% |
| synth_d7_s0 | synthetic | 53 | Quiesced → Quiesced | 51 → 51 | 119 → 119 | 20216 → 20216 | 518 → 518 | 724 → 724 | +0.00% |
| synth_d7_s1 | synthetic | 166 | Quiesced → Quiesced | 140 → 140 | 300 → 300 | 50904 → 50904 | 1320 → 1320 | 2583 → 2583 | +0.00% |
| synth_d7_s2 | synthetic | 6 | Quiesced → Quiesced | 6 → 6 | 8 → 8 | 1344 → 1344 | 5 → 5 | 94 → 94 | +0.00% |
| synth_d7_s3 | synthetic | 186 | Quiesced → Quiesced | 186 → 186 | 603 → 603 | 102872 → 102872 | 3740 → 3740 | 2198 → 2198 | +0.00% |
| synth_d7_s4 | synthetic | 98 | Quiesced → Quiesced | 111 → 111 | 498 → 498 | 83944 → 83944 | 4539 → 4539 | 1419 → 1419 | +0.00% |
| synth_d7_s5 | synthetic | 54 | Quiesced → Quiesced | 66 → 66 | 370 → 370 | 63952 → 63952 | 3013 → 3013 | 652 → 652 | +0.00% |
| synth_d7_s6 | synthetic | 95 | Quiesced → Quiesced | 99 → 99 | 341 → 341 | 58520 → 58520 | 2142 → 2142 | 1454 → 1454 | +0.00% |
| synth_d7_s7 | synthetic | 85 | Quiesced → Quiesced | 138 → 138 | 777 → 777 | 131600 → 131600 | 8082 → 8082 | 802 → 802 | +0.00% |
| synth_d7_s8 | synthetic | 122 | ClassCap(Allocated) → ClassCap(Live) | 422 → 561 | 3453 → 5611 | 586544 → 953120 | 11753 → 22688 | 1387 → 1387 | +0.00% |
| synth_d7_s9 | synthetic | 14 | Quiesced → Quiesced | 10 → 10 | 18 → 18 | 3136 → 3136 | 21 → 21 | 147 → 147 | +0.00% |
| synth_d7_s10 | synthetic | 58 | Quiesced → Quiesced | 43 → 43 | 100 → 100 | 17304 → 17304 | 368 → 368 | 413 → 413 | +0.00% |
| synth_d7_s11 | synthetic | 73 | ClassCap(Allocated) → ClassCap(Live) | 483 → 572 | 3532 → 5496 | 604352 → 940072 | 10613 → 20654 | 970 → 970 | +0.00% |
| synth_d7_s13 | synthetic | 217 | ClassCap(Allocated) → ClassCap(Live) | 878 → 1147 | 4324 → 6089 | 735784 → 1036336 | 11144 → 15284 | 2386 → 2386 | +0.00% |
| synth_d7_s14 | synthetic | 72 | ClassCap(Allocated) → ClassCap(Live) | 414 → 615 | 3812 → 6736 | 652400 → 1151248 | 12408 → 26878 | 680 → 680 | +0.00% |
| synth_d7_s15 | synthetic | 56 | Quiesced → Quiesced | 45 → 45 | 74 → 74 | 12656 → 12656 | 78 → 78 | 709 → 709 | +0.00% |
| synth_d7_s16 | synthetic | 36 | Quiesced → Quiesced | 26 → 26 | 57 → 57 | 9688 → 9688 | 144 → 144 | 249 → 249 | +0.00% |
| synth_d7_s17 | synthetic | 137 | ClassCap(Allocated) → ClassCap(Live) | 319 → 343 | 2229 → 2930 | 377720 → 495488 | 61015 → 107314 | 1803 → 1803 | +0.00% |
| synth_d7_s18 | synthetic | 203 | Quiesced → Quiesced | 436 → 436 | 2839 → 2839 | 479528 → 479528 | 51016 → 51016 | 2943 → 2943 | +0.00% |
| synth_d7_s19 | synthetic | 105 | ClassCap(Allocated) → Quiesced | 327 → 320 | 2881 → 3148 | 492352 → 537936 | 20751 → 62376 | 1931 → 1931 | +0.00% |
| synth_d7_s20 | synthetic | 106 | Quiesced → Quiesced | 103 → 103 | 371 → 371 | 63504 → 63504 | 2383 → 2383 | 1440 → 1440 | +0.00% |
| synth_d7_s21 | synthetic | 81 | Quiesced → Quiesced | 59 → 59 | 140 → 140 | 24136 → 24136 | 675 → 675 | 612 → 612 | +0.00% |
| synth_d7_s22 | synthetic | 45 | Quiesced → Quiesced | 55 → 55 | 270 → 270 | 46200 → 46200 | 1880 → 1880 | 608 → 608 | +0.00% |
| synth_d7_s23 | synthetic | 122 | ClassCap(Allocated) → ClassCap(Live) | 492 → 535 | 3584 → 6133 | 610232 → 1047032 | 10228 → 29908 | 2351 → 2351 | +0.00% |
| synth_d7_s24 | synthetic | 62 | Quiesced → Quiesced | 48 → 48 | 85 → 85 | 14560 → 14560 | 212 → 212 | 674 → 674 | +0.00% |
| synth_d7_s25 | synthetic | 14 | Quiesced → Quiesced | 14 → 14 | 21 → 21 | 3528 → 3528 | 17 → 17 | 123 → 123 | +0.00% |
| synth_d7_s26 | synthetic | 123 | ClassCap(Allocated) → ClassCap(Live) | 994 → 2258 | 4650 → 9828 | 784168 → 1666336 | 9732 → 18110 | 1786 → 1786 | +0.00% |
| synth_d7_s27 | synthetic | 55 | Quiesced → Quiesced | 42 → 42 | 83 → 83 | 14280 → 14280 | 251 → 251 | 762 → 762 | +0.00% |
| synth_d7_s28 | synthetic | 146 | ClassCap(Allocated) → ClassCap(Live) | 410 → 446 | 3228 → 5309 | 565824 → 921872 | 8177 → 24831 | 2311 → 2311 | +0.00% |
| synth_d7_s29 | synthetic | 99 | Quiesced → Quiesced | 113 → 113 | 468 → 468 | 80024 → 80024 | 3736 → 3736 | 1438 → 1438 | +0.00% |
| synth_d7_s30 | synthetic | 83 | Quiesced → Quiesced | 70 → 70 | 157 → 157 | 26544 → 26544 | 483 → 483 | 1351 → 1351 | +0.00% |
| synth_d7_s31 | synthetic | 135 | ClassCap(Allocated) → ClassCap(Live) | 740 → 1223 | 4189 → 8717 | 725536 → 1517264 | 7598 → 20106 | 1582 → 1582 | +0.00% |
| synth_d7_s32 | synthetic | 144 | Quiesced → Quiesced | 136 → 136 | 385 → 385 | 65240 → 65240 | 1881 → 1881 | 2026 → 2026 | +0.00% |
| synth_d7_s33 | synthetic | 46 | Quiesced → Quiesced | 40 → 40 | 68 → 68 | 11592 → 11592 | 123 → 123 | 512 → 512 | +0.00% |
| synth_d7_s34 | synthetic | 37 | Quiesced → Quiesced | 34 → 34 | 59 → 59 | 10192 → 10192 | 103 → 103 | 503 → 503 | +0.00% |
| synth_d7_s35 | synthetic | 139 | Quiesced → Quiesced | 186 → 186 | 1187 → 1187 | 202944 → 202944 | 14009 → 14009 | 1832 → 1832 | +0.00% |
| synth_d7_s36 | synthetic | 72 | Quiesced → Quiesced | 75 → 75 | 234 → 234 | 39760 → 39760 | 2058 → 2058 | 729 → 729 | +0.00% |
| synth_d7_s37 | synthetic | 143 | Quiesced → Quiesced | 108 → 108 | 188 → 188 | 32256 → 32256 | 372 → 372 | 2441 → 2441 | +0.00% |
| synth_d7_s38 | synthetic | 132 | Quiesced → Quiesced | 137 → 137 | 658 → 658 | 113624 → 113624 | 5108 → 5108 | 1410 → 1410 | +0.00% |
| synth_d7_s39 | synthetic | 43 | Quiesced → Quiesced | 55 → 55 | 266 → 266 | 45528 → 45528 | 2258 → 2258 | 659 → 659 | +0.00% |
| synth_d9_s0 | synthetic | 318 | ClassCap(Allocated) → ClassCap(Live) | 867 → 1644 | 3541 → 8066 | 598976 → 1371888 | 10119 → 20919 | 4257 → 4257 | +0.00% |
| synth_d9_s1 | synthetic | 83 | Quiesced → Quiesced | 67 → 67 | 137 → 137 | 23296 → 23296 | 546 → 546 | 1235 → 1235 | +0.00% |
| synth_d9_s2 | synthetic | 84 | Quiesced → Quiesced | 69 → 69 | 121 → 121 | 20440 → 20440 | 322 → 322 | 1055 → 1055 | +0.00% |
| synth_d9_s3 | synthetic | 300 | Quiesced → Quiesced | 361 → 361 | 1689 → 1689 | 288512 → 288512 | 18945 → 18945 | 4198 → 4198 | +0.00% |
| synth_d9_s4 | synthetic | 47 | Quiesced → Quiesced | 35 → 35 | 68 → 68 | 11648 → 11648 | 153 → 153 | 668 → 668 | +0.00% |
| synth_d9_s5 | synthetic | 213 | Quiesced → Quiesced | 169 → 169 | 345 → 345 | 58744 → 58744 | 1051 → 1051 | 3551 → 3551 | +0.00% |
| synth_d9_s6 | synthetic | 156 | Quiesced → Quiesced | 121 → 121 | 239 → 239 | 40768 → 40768 | 765 → 765 | 2039 → 2039 | +0.00% |
| synth_d9_s7 | synthetic | 24 | Quiesced → Quiesced | 19 → 19 | 30 → 30 | 5208 → 5208 | 31 → 31 | 459 → 459 | +0.00% |
| synth_d9_s8 | synthetic | 188 | ClassCap(Allocated) → ClassCap(Live) | 1088 → 1474 | 4320 → 6861 | 728952 → 1156400 | 7464 → 10584 | 1946 → 1946 | +0.00% |
| synth_d9_s9 | synthetic | 41 | Quiesced → Quiesced | 38 → 38 | 66 → 66 | 11200 → 11200 | 129 → 129 | 253 → 253 | +0.00% |
| synth_d9_s10 | synthetic | 184 | Quiesced → Quiesced | 202 → 202 | 842 → 842 | 143024 → 143024 | 8234 → 8234 | 2381 → 2381 | +0.00% |
| synth_d9_s11 | synthetic | 38 | Quiesced → Quiesced | 28 → 28 | 50 → 50 | 8680 → 8680 | 57 → 57 | 317 → 317 | +0.00% |
| synth_d9_s12 | synthetic | 104 | Quiesced → Quiesced | 91 → 91 | 171 → 171 | 29120 → 29120 | 432 → 432 | 1577 → 1577 | +0.00% |
| synth_d9_s14 | synthetic | 69 | Quiesced → Quiesced | 50 → 50 | 99 → 99 | 17024 → 17024 | 298 → 298 | 1053 → 1053 | +0.00% |
| synth_d9_s15 | synthetic | 151 | Quiesced → Quiesced | 135 → 135 | 304 → 304 | 51856 → 51856 | 1256 → 1256 | 1954 → 1954 | +0.00% |
| synth_d9_s16 | synthetic | 111 | Quiesced → Quiesced | 103 → 103 | 224 → 224 | 38192 → 38192 | 848 → 848 | 1304 → 1304 | +0.00% |
| synth_d9_s17 | synthetic | 15 | Quiesced → Quiesced | 9 → 9 | 20 → 20 | 3528 → 3528 | 57 → 57 | 162 → 162 | +0.00% |
| synth_d9_s18 | synthetic | 15 | Quiesced → Quiesced | 10 → 10 | 18 → 18 | 3136 → 3136 | 21 → 21 | 282 → 282 | +0.00% |
| synth_d9_s19 | synthetic | 109 | Quiesced → Quiesced | 123 → 123 | 470 → 470 | 80192 → 80192 | 3102 → 3102 | 1928 → 1928 | +0.00% |
| synth_d9_s20 | synthetic | 147 | Quiesced → Quiesced | 113 → 113 | 200 → 200 | 33936 → 33936 | 376 → 376 | 2004 → 2004 | +0.00% |
| synth_d9_s21 | synthetic | 106 | Quiesced → Quiesced | 79 → 79 | 163 → 163 | 27832 → 27832 | 621 → 621 | 1718 → 1718 | +0.00% |
| synth_d9_s22 | synthetic | 101 | Quiesced → Quiesced | 103 → 103 | 275 → 275 | 46928 → 46928 | 2594 → 2594 | 1133 → 1133 | +0.00% |
| synth_d9_s23 | synthetic | 51 | Quiesced → Quiesced | 34 → 34 | 64 → 64 | 11144 → 11144 | 80 → 80 | 442 → 442 | +0.00% |
| synth_d9_s24 | synthetic | 206 | ClassCap(Allocated) → ClassCap(Live) | 489 → 647 | 3613 → 5904 | 614656 → 1004192 | 12456 → 24502 | 3157 → 3157 | +0.00% |
| synth_d9_s26 | synthetic | 73 | Quiesced → Quiesced | 84 → 84 | 205 → 205 | 34776 → 34776 | 709 → 709 | 1010 → 1010 | +0.00% |
| synth_d9_s27 | synthetic | 113 | Quiesced → Quiesced | 103 → 103 | 201 → 201 | 34048 → 34048 | 708 → 708 | 1158 → 1158 | +0.00% |
| synth_d9_s28 | synthetic | 15 | Quiesced → Quiesced | 10 → 10 | 19 → 19 | 3304 → 3304 | 23 → 23 | 96 → 96 | +0.00% |
| synth_d9_s30 | synthetic | 111 | Quiesced → Quiesced | 90 → 90 | 178 → 178 | 30296 → 30296 | 545 → 545 | 1957 → 1957 | +0.00% |
| synth_d9_s31 | synthetic | 120 | Quiesced → Quiesced | 88 → 88 | 172 → 172 | 29400 → 29400 | 634 → 634 | 1575 → 1575 | +0.00% |
| synth_d9_s32 | synthetic | 318 | ClassCap(Allocated) → ClassCap(Live) | 818 → 1241 | 4405 → 7763 | 751408 → 1321544 | 12396 → 19369 | 4175 → 4175 | +0.00% |
| synth_d9_s33 | synthetic | 77 | Quiesced → Quiesced | 56 → 56 | 103 → 103 | 17752 → 17752 | 216 → 216 | 1434 → 1434 | +0.00% |
| synth_d9_s34 | synthetic | 103 | Quiesced → Quiesced | 114 → 114 | 385 → 385 | 65800 → 65800 | 2467 → 2467 | 1521 → 1521 | +0.00% |
| synth_d9_s35 | synthetic | 267 | Quiesced → Quiesced | 292 → 292 | 1326 → 1326 | 226016 → 226016 | 14709 → 14709 | 3098 → 3098 | +0.00% |
| synth_d9_s36 | synthetic | 156 | Quiesced → Quiesced | 219 → 219 | 1199 → 1199 | 204736 → 204736 | 13428 → 13428 | 1142 → 1142 | +0.00% |
| synth_d9_s37 | synthetic | 176 | Quiesced → Quiesced | 181 → 181 | 919 → 919 | 156688 → 156688 | 12874 → 12874 | 2419 → 2419 | +0.00% |
| synth_d9_s38 | synthetic | 104 | Quiesced → Quiesced | 78 → 78 | 138 → 138 | 23632 → 23632 | 261 → 261 | 1622 → 1622 | +0.00% |
| synth_d9_s39 | synthetic | 179 | Quiesced → Quiesced | 135 → 135 | 282 → 282 | 47880 → 47880 | 1093 → 1093 | 2296 → 2296 | +0.00% |
| synth_d11_s0 | synthetic | 279 | Quiesced → Quiesced | 234 → 234 | 582 → 582 | 98728 → 98728 | 3263 → 3263 | 4028 → 4028 | +0.00% |
| synth_d11_s1 | synthetic | 206 | ClassCap(Allocated) → ClassCap(Live) | 660 → 818 | 3998 → 6600 | 681352 → 1126048 | 9008 → 20018 | 2962 → 2962 | +0.00% |
| synth_d11_s2 | synthetic | 19 | Quiesced → Quiesced | 12 → 12 | 26 → 26 | 4536 → 4536 | 64 → 64 | 147 → 147 | +0.00% |
| synth_d11_s4 | synthetic | 89 | Quiesced → Quiesced | 89 → 89 | 232 → 232 | 39648 → 39648 | 2176 → 2176 | 1665 → 1665 | +0.00% |
| synth_d11_s5 | synthetic | 456 | ClassCap(Allocated) → ClassCap(Live) | 973 → 2637 | 4588 → 10095 | 775712 → 1703632 | 11292 → 18860 | 6816 → 6816 | +0.00% |
| synth_d11_s6 | synthetic | 95 | Quiesced → Quiesced | 75 → 75 | 133 → 133 | 22736 → 22736 | 262 → 262 | 1754 → 1754 | +0.00% |
| synth_d11_s8 | synthetic | 372 | ClassCap(Allocated) → ClassCap(Live) | 817 → 1098 | 3983 → 7812 | 687456 → 1346576 | 8809 → 21059 | 4916 → 4916 | +0.00% |
| synth_d11_s9 | synthetic | 21 | Quiesced → Quiesced | 18 → 18 | 27 → 27 | 4592 → 4592 | 25 → 25 | 372 → 372 | +0.00% |
| synth_d11_s10 | synthetic | 22 | Quiesced → Quiesced | 20 → 20 | 33 → 33 | 5600 → 5600 | 50 → 50 | 345 → 345 | +0.00% |
| synth_d11_s11 | synthetic | 89 | Quiesced → Quiesced | 72 → 72 | 159 → 159 | 27160 → 27160 | 556 → 556 | 1338 → 1338 | +0.00% |
| synth_d11_s12 | synthetic | 329 | ClassCap(Allocated) → ClassCap(Live) | 483 → 525 | 2401 → 2614 | 404096 → 440048 | 10207 → 11381 | 5020 → 5020 | +0.00% |
| synth_d11_s13 | synthetic | 25 | Quiesced → Quiesced | 25 → 25 | 47 → 47 | 8064 → 8064 | 87 → 87 | 195 → 195 | +0.00% |
| synth_d11_s14 | synthetic | 636 | ClassCap(Allocated) → ClassCap(Live) | 1219 → 1520 | 4463 → 6496 | 759864 → 1103536 | 8073 → 11412 | 7675 → 7675 | +0.00% |
| synth_d11_s15 | synthetic | 181 | ClassCap(Allocated) → ClassCap(Live) | 749 → 1304 | 4369 → 9067 | 744856 → 1557472 | 11351 → 19605 | 2975 → 2975 | +0.00% |
| synth_d11_s16 | synthetic | 135 | ClassCap(Allocated) → ClassCap(Live) | 743 → 1154 | 4151 → 9557 | 706160 → 1625064 | 9448 → 26412 | 1729 → 1729 | +0.00% |
| synth_d11_s17 | synthetic | 142 | Quiesced → Quiesced | 122 → 122 | 237 → 237 | 40320 → 40320 | 723 → 723 | 2177 → 2177 | +0.00% |
| synth_d11_s18 | synthetic | 221 | ClassCap(Allocated) → ClassCap(Live) | 363 → 438 | 1951 → 2715 | 329840 → 458360 | 8389 → 14671 | 2144 → 2144 | +0.00% |
| synth_d11_s19 | synthetic | 202 | Quiesced → Quiesced | 191 → 191 | 585 → 585 | 99792 → 99792 | 3661 → 3661 | 2353 → 2353 | +0.00% |
| synth_d11_s20 | synthetic | 252 | ClassCap(Allocated) → ClassCap(Live) | 1173 → 1500 | 4543 → 7653 | 780192 → 1317624 | 7721 → 11952 | 3283 → 3283 | +0.00% |
| synth_d11_s21 | synthetic | 80 | Quiesced → Quiesced | 52 → 52 | 99 → 99 | 16912 → 16912 | 243 → 243 | 905 → 905 | +0.00% |
| synth_d11_s22 | synthetic | 366 | Quiesced → Quiesced | 472 → 472 | 2336 → 2336 | 399168 → 399168 | 50981 → 50981 | 5564 → 5564 | +0.00% |
| synth_d11_s24 | synthetic | 495 | ClassCap(Allocated) → ClassCap(Live) | 958 → 1112 | 4373 → 5996 | 745640 → 1024800 | 9363 → 13046 | 7514 → 7514 | +0.00% |
| synth_d11_s25 | synthetic | 86 | Quiesced → Quiesced | 64 → 64 | 128 → 128 | 21840 → 21840 | 366 → 366 | 1074 → 1074 | +0.00% |
| synth_d11_s26 | synthetic | 285 | Quiesced → Quiesced | 299 → 299 | 903 → 903 | 153216 → 153216 | 6360 → 6360 | 3607 → 3607 | +0.00% |
| synth_d11_s27 | synthetic | 289 | ClassCap(Allocated) → Quiesced | 420 → 416 | 2859 → 3112 | 485464 → 528528 | 22185 → 58852 | 4665 → 4665 | +0.00% |
| synth_d11_s28 | synthetic | 114 | Quiesced → Quiesced | 84 → 84 | 144 → 144 | 24584 → 24584 | 365 → 365 | 1419 → 1419 | +0.00% |
| synth_d11_s29 | synthetic | 808 | ClassCap(Allocated) → ClassCap(Live) | 952 → 1097 | 3948 → 6358 | 673008 → 1086176 | 11331 → 29152 | 10296 → 10296 | +0.00% |
| synth_d11_s30 | synthetic | 23 | Quiesced → Quiesced | 25 → 25 | 54 → 54 | 9128 → 9128 | 125 → 125 | 201 → 201 | +0.00% |
| synth_d11_s31 | synthetic | 279 | ClassCap(Allocated) → ClassCap(Live) | 924 → 1448 | 4514 → 7677 | 781480 → 1321432 | 8081 → 12910 | 3763 → 3763 | +0.00% |
| synth_d11_s32 | synthetic | 414 | ClassCap(Allocated) → ClassCap(Live) | 629 → 693 | 3408 → 5003 | 583016 → 854840 | 9263 → 17444 | 5595 → 5595 | +0.00% |
| synth_d11_s33 | synthetic | 194 | Quiesced → Quiesced | 216 → 216 | 851 → 851 | 145208 → 145208 | 6870 → 6870 | 2827 → 2827 | +0.00% |
| synth_d11_s35 | synthetic | 238 | Quiesced → Quiesced | 184 → 184 | 377 → 377 | 63952 → 63952 | 1179 → 1179 | 2921 → 2921 | +0.00% |
| synth_d11_s36 | synthetic | 24 | Quiesced → Quiesced | 15 → 15 | 29 → 29 | 5040 → 5040 | 37 → 37 | 303 → 303 | +0.00% |
| synth_d11_s37 | synthetic | 216 | ClassCap(Allocated) → ClassCap(Live) | 864 → 1062 | 4348 → 6532 | 737464 → 1109696 | 8673 → 12691 | 3496 → 3496 | +0.00% |
| synth_d11_s38 | synthetic | 230 | Quiesced → Quiesced | 260 → 260 | 1181 → 1181 | 200648 → 200648 | 11296 → 11296 | 3418 → 3418 | +0.00% |
| synth_d9_s29 | synthetic | 371 | ClassCap(Allocated) → ClassCap(Live) | 1088 → 1861 | 4609 → 8710 | 787584 → 1480640 | 8474 → 14772 | 4999 → 5000 | -0.02% |
| synth_d11_s7 | synthetic | 454 | ClassCap(Allocated) → ClassCap(Live) | 624 → 784 | 3298 → 5834 | 563304 → 993832 | 8025 → 24483 | 7254 → 7258 | -0.06% |
| synth_d9_s25 | synthetic | 764 | ClassCap(Allocated) → ClassCap(Live) | 1091 → 1963 | 3539 → 7753 | 598472 → 1309280 | 7003 → 16993 | 10496 → 10502 | -0.06% |
| synth_d11_s34 | synthetic | 265 | ClassCap(Allocated) → ClassCap(Live) | 654 → 771 | 3678 → 7006 | 624008 → 1193752 | 9023 → 30398 | 3781 → 3786 | -0.13% |
| synth_d7_s12 | synthetic | 93 | ClassCap(Allocated) → ClassCap(Live) | 340 → 343 | 2462 → 2850 | 415296 → 480592 | 10001 → 11199 | 1112 → 1115 | -0.27% |
| shader:julia_set | shader | 122 | ClassCap(Allocated) → ClassCap(Live) | 819 → 1402 | 4533 → 8127 | 775544 → 1388464 | 14866 → 20564 | 716 → 728 | -1.68% |
| synth_d9_s13 | synthetic | 297 | ClassCap(Allocated) → ClassCap(Live) | 723 → 1205 | 3543 → 7214 | 600600 → 1240624 | 8011 → 19449 | 4618 → 4770 | -3.29% |
| psychedelic | psychedelic | 102 | ClassCap(Allocated) → ClassCap(Live) | 946 → 2561 | 4847 → 9885 | 822528 → 1669584 | 7843 → 14937 | 766 → 807 | -5.35% |
| glyph16:U+004B | glyph16 | 617 | ClassCap(Allocated) → ClassCap(Live) | 925 → 1458 | 4578 → 7748 | 783048 → 1329328 | 10063 → 17070 | 1047 → 1121 | -7.07% |
| glyph32:U+004B | glyph32 | 617 | ClassCap(Allocated) → ClassCap(Live) | 925 → 1458 | 4578 → 7748 | 783048 → 1329328 | 10063 → 17070 | 1047 → 1121 | -7.07% |
