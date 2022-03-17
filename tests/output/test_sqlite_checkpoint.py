import pytest
from firedrake import *
import numpy as np


def test_store_load():
    mesh = UnitSquareMesh(2, 2)
    degree = 1
    family = "CG"
    V = FunctionSpace(mesh, family, degree)

    f = Function(V, name="f")
    x = SpatialCoordinate(mesh)

    f.interpolate(x[0] * x[1])

    f2 = Function(V, name="f")

    with SQLiteCheckpoint(":memory:", mode="rw") as chk:
        chk.store(f)
        chk.load(f2)

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_store_read_only_ioerror():
    mesh = UnitSquareMesh(2, 2)
    degree = 1
    family = "CG"
    V = FunctionSpace(mesh, family, degree)
    f = Function(V, name="f")

    with SQLiteCheckpoint(":memory:", "r") as chk:
        with pytest.raises(IOError):
            chk.store(f)
