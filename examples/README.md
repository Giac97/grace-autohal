# Examples

| Folder | System |
|---|---|
| `pd111_co/` | Pd(111) 3×3×5 + CO, 38 seed structures, bwUniCluster profile |
| `al_bulk/` | fcc Al 2×2×2 with vacancies, bwUniCluster profile |

Copy a folder, set `aims.bin`, `aims.species_dir` and `hpc.partition_gpu`, then
`grace-al prepare config.yaml --seeds` and `grace-al submit config.yaml`.
