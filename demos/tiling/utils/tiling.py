from pyop2.mpi import MPI


def calculate_sdepth(num_solves, num_unroll, extra_halo):
    """The sdepth is calculated through the following formula:

        sdepth = 1 if sequential else 1 + num_solves*num_unroll + extra_halo

    Where:

    :arg num_solves: number of solves per loop chain iteration
    :arg num_unroll: unroll factor for the loop chain
    :arg extra_halo: to expose the nonexec region to the tiling engine
    """
    if MPI.parallel:
        return 1 + num_solves*num_unroll + extra_halo
    else:
        return 1