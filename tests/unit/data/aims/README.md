FHI-aims outputs for the parser tests.

| file | source |
|---|---|
| ok_slab.out | Pd(111) 3x3x5 + O, bwUniCluster 3.0, aims 261007, Intel MPI (SCF iterations 1-66 cut out) |
| ok_bulk.out | Al fcc bulk, 1 atom, workstation aims 260331, OpenMPI (iterations 1-9 cut out) |
| scf_fail.out/.err | same Al bulk with `sc_iter_limit 3` |
| setup_fail.out/.err | control.in without species |
| charge_integration.out/.err | Pd slab, bwUniCluster before the pdgemm / I_MPI fix (trimmed) |
| interrupted.out | Pd slab killed by the job time limit after 33 iterations (trimmed) |
| stacksize.out | synthetic: ok_bulk header + the messages of check_stacksize.f90 |
| mpi.out/.err | synthetic: empty aims.out, OpenMPI's prterun error |
