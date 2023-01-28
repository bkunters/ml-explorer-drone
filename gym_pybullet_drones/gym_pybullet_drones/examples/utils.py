from mpi4py import MPI

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()