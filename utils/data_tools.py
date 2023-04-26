from utils.load_data import canonical_distances_and_dihedrals, extract_backbone_coords, extract_backbone_seqs, EXHAUSTIVE_DISTS, EXHAUSTIVE_ANGLES
from biotite.structure.io.pdb import PDBFile
import gzip
import biotite.structure as struc
import numpy as np
import pandas as pd
from utils import modulo_with_wrapped_range

alphabet='ACDEFGHIKLMNPQRSTVWYU'
alphabet_map = {val:i for i,val in enumerate(alphabet)}

class ReadPDB:
    def __init__(self):
        pass
    
    @classmethod
    def read_pdb(self, fname, angles=EXHAUSTIVE_ANGLES, atoms=["N", "CA", "C"]):
        if fname[-7:] == ".pdb.gz":
            source = PDBFile.read(gzip.open(str(fname), mode="rt"))
        else:
            source = PDBFile.read(str(fname))
        
        
        if source.get_model_count() > 1:
            return None

        source_struct = source.get_structure(extra_fields=["b_factor"])[0]
        
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
        
        # Get any additional angles
        non_dihedral_angles = [a for a in angles if a not in calc_angles]
        backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
        
        for a in non_dihedral_angles:
            if a == "tau" or a == "N:CA:C":
                # tau = N - CA - C internal angles
                idx = np.array(
                    [list(range(i, i + 3)) for i in range(3, len(backbone_atoms), 3)]
                    + [(0, 0, 0)]
                )
            elif a == "CA:C:1N":  # Same as C-N angle in nerf
                # This measures an angle between two residues. Due to the way we build
                # proteins out later, we do not need to meas
                idx = np.array(
                    [(i + 1, i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                    + [(0, 0, 0)]
                )
            elif a == "C:1N:1CA":
                idx = np.array(
                    [(i + 2, i + 3, i + 4) for i in range(0, len(backbone_atoms) - 3, 3)]
                    + [(0, 0, 0)]
                )
            else:
                raise ValueError(f"Unrecognized angle: {a}")
            calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx)
            
        struct_arrays = pd.DataFrame({k: calc_angles[k].squeeze() for k in angles})
        
        
        
        
        ca = [c for c in backbone_atoms if c.atom_name in atoms]
        coord_arrays = np.vstack([c.coord for c in ca])
        
        
        
        AA_NAME_SYM = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': "U"
        }
        ca = [c for c in backbone_atoms if c.atom_name in ["CA"]]
        index = [(one.chain_id, one.res_id) for one in ca ]
        attn_mask = np.zeros(len(ca)) == 0
        b_factor = np.zeros(len(ca))
        amino_acids = []
        for idx, c in enumerate(ca):
            if c.atom_name in atoms:
                if AA_NAME_SYM.get(c.res_name) is not None:
                    amino_acids.append(AA_NAME_SYM[c.res_name])
                else:
                    amino_acids.append("U")
            
            if c.b_factor == 0:
                attn_mask[idx] = False
            b_factor[idx] = c.b_factor
        
        angles = modulo_with_wrapped_range(struct_arrays.iloc[:,[0,1,2,3,4,5]], -np.pi, np.pi).values
        np.nan_to_num(angles, copy=False, nan=0)
        coords = coord_arrays.reshape(angles.shape[0],3,3)
        seqs = np.array([alphabet_map[one] for one in amino_acids]).reshape(-1,1)
        
        return index, angles, coords, seqs, attn_mask, b_factor