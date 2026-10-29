# bwUniCluster 3.0 (KIT)

`site: bwunicluster3` (or `grace-al init ... --site bwunicluster3`) sets:

```yaml
aims:
  parameters:
    elsi_build_dm_method: pdgemm      # the Intel 2025 build's default DM builder is wrong
hpc:
  partition_gpu: gpu_h100             # placeholders: check yours with `grace-al prepare`
  partition_cpu: cpu
  gres: gpu:1
  ntasks: 32
  mem: 128G
  time: "48:00:00"
  modules: [compiler/intel/2025.0, mpi/impi/2021.14-intel-2025.0, numlib/mkl/2025.0]
  env:                                # Intel MPI on the AMD nodes (shared node)
    I_MPI_COLL_INTRANODE: pt2pt
    I_MPI_PIN_DOMAIN: "1"
    I_MPI_PIN_CELL: core
  activate: eval "$(micromamba shell hook --shell bash)"; micromamba activate grace
  mpirun: mpirun
```

## Partitions

Which partitions you may use depends on your group (`id -Gn`) and account. `grace-al prepare`
prints them; by hand:

```bash
for p in $(sinfo -h -o %R | sort -u); do
  echo "$p: $(scontrol show partition $p | grep -oE '(AllowGroups|AllowAccounts|MaxTime|TRES)=[^ ]+' | tr '\n' ' ')"
done; id -Gn
```

Put the GPU partition open to you in `hpc.partition_gpu` (your config wins over the profile).

## FHI-aims build notes

- Build aims with the Intel compiler, Intel MPI and MKL modules above; `aims.bin` points to
  `aims.x` (`$HOME/FHIaims-master/build/aims.x` in the templates).
- **Charge integration errors** with this build come from two things, both handled by the
  profile: ELSI's default density-matrix builder (ELPA A^T·A) gives a wrong density matrix
  (→ `elsi_build_dm_method pdgemm`), and Intel MPI's shared-memory collectives with loose
  pinning on partially used AMD nodes (→ the three `I_MPI_*` variables).
- `I_MPI_DEBUG=5` (in `hpc.env`) prints the pinning table at the top of `aims.out`.
- The Python env ships its own Intel MPI. The job loads the modules first and saves that
  environment (`runs/.aims_env`); every aims run gets it back.
