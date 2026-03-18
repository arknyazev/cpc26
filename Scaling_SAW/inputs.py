from dataclasses import dataclass
import numpy as np

@dataclass
class Inputs:
    boozmn_filename: str
    ic_file: str
    saw_filename: str
    nParticles: int
    resolution: int
    tol: float
    tmax: float

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(
            boozmn_filename=str(d["boozmn_filename"]),
            ic_file=str(d["ic_file"]),
            saw_filename=str(d["saw_filename"]),
            nParticles=int(d["nParticles"]),
            resolution=int(d["resolution"]),
            tol=float(d["tol"]),
            tmax=float(d["tmax"]),
        )

    def to_npz(self, path):
        np.savez(
            path,
            boozmn_filename=self.boozmn_filename,
            ic_file=self.ic_file,
            saw_filename=self.saw_filename,
            nParticles=self.nParticles,
            resolution=self.resolution,
            tol=self.tol,
            tmax=self.tmax,
        )
