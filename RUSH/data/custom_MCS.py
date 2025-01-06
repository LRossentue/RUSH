from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdFMCS import BondCompare, RingCompare
import re


def _fix_querymol(smarts):
    return Chem.MolFromSmarts(''.join(
        "[#0,!D1]" if ('#0' in _) else f"[{_}]" if ('#' in _)
        else _ for _ in re.split(r"[\[\]]", smarts)))


def _get_MCS_res(mol, fragment, avoid_indices=None, allowance=0.9):
    # must be initialized for each molecule to reset flags and global params.
    params = rdFMCS.MCSParameters()
    # custom overwrites, requires rdkit v.2023.+
    mcs_acceptance = CustomMCSAcceptance(allowance=allowance)
    params.ShouldAcceptMCS = mcs_acceptance
    # changed from Atom to AtomTyper (renamed in official rdkit rollout)
    params.AtomTyper = CustomCompareElements(avoid_indices=avoid_indices)

    params.BondTyper = BondCompare.CompareOrderExact
    params.RingTyper = RingCompare.PermissiveRingFusion
    # actually True, handled in custom call instead.
    params.AtomCompareParameters.RingMatchesRingOnly = False
    params.AtomCompareParameters.CompleteRingsOnly = False
    params.AtomCompareParameters.MatchValences = True
    params.AtomCompareParameters.MatchChiralTag = False
    params.BondCompareParameters.MatchFusedRings = True
    params.BondCompareParameters.MatchFusedRingsStrict = False
    params.Timeout = 1
    params.MaximizeBonds = True

    # find the best MCS that contains (A) and has at least allowance * n atoms.
    res = rdFMCS.FindMCS([mol, fragment], params)
    # if flag, set res to None instead.
    if not mcs_acceptance.mcs_found:
        res = None
    return res


class CustomCompareElements(rdFMCS.MCSAtomCompare):
    def __init__(self, avoid_indices=None):
        super().__init__()
        # A list of atom indices that are not allowed to be in an MCS result.
        # Used to identify several MCSs in a given molecule for multiple ...
        # consecutive queries that can not overlap.
        self.avoid_indices = avoid_indices if avoid_indices else []

    # edited atom compare element for rdFMCS to also match the (A) dummy atom.
    # (A) atom must have a degree of 1 (attachment point of the query fragment).
    # Atom match for (A) must have a degree greater than 1 (not a peripheral atom).

    def __call__(self, p, mol1, atom1, mol2, atom2):
        # figure out which is which, assuming mol is larger than fragment.
        mol_atom, fragment_atom = (atom1, atom2) if (len(mol1.GetAtoms()) >
                                                     len(mol2.GetAtoms())) else (atom2, atom1)
        # if the atom is not allowed, break out
        if mol_atom in self.avoid_indices:
            return False

        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        a1_is_dummy = (a1.GetAtomicNum() == 0)
        a2_is_dummy = (a2.GetAtomicNum() == 0)

        if a1_is_dummy ^ a2_is_dummy:
            atoms = (a1, a2)
            dummy_atom_idx = 0 if a1_is_dummy else 1
            other_atom_idx = 1 - dummy_atom_idx
            return (atoms[dummy_atom_idx].GetDegree() == 1 and
                    atoms[other_atom_idx].GetDegree() > 1)

        if (a1.GetAtomicNum() != a2.GetAtomicNum()):
            return False

        return self.CheckAtomRingMatch(p,mol1, atom1, mol2, atom2)


class CustomMCSAcceptance(rdFMCS.MCSAcceptance):
    # edited MCS acceptance element for rdFMCS to only accept an MCS if it:
    # 1) contains the (A) dummy atom in the match (see CustomCompareElements above).
    # 2) is at least self.allowance * atom count in size.
    # this, because we want to identify (sufficiently) complete fragments in mol only.
    # we assume the provided mol is always larger in atom count than the fragment.
    def __init__(self, allowance):
        super().__init__()

        self._allowance = allowance
        self._mcs_found = False

    @property
    def mcs_found(self):
        return self._mcs_found

    def __call__(self, query, target, match, match_bonds, params):
        # resolve which is which, assuming mol is larger.
        query_indices, target_indices = tuple(zip(*match))
        mol, fragment, mol_indices, fragment_indices = (query,
                                                        target,
                                                        query_indices,
                                                        target_indices) if (
                      len(query.GetAtoms()) > len(target.GetAtoms())) else (
                                                        target,
                                                        query,
                                                        target_indices,
                                                        query_indices
        )
        mol_anchor = None
        # evaluate if anchor in the match, and retrieve the equivalent atom in mol.
        for m_i, f_i in zip(mol_indices, fragment_indices):
            f_atom = fragment.GetAtomWithIdx(int(f_i))
            if f_atom.GetAtomicNum() == 0:
                mol_anchor = mol.GetAtomWithIdx(int(m_i))
                break

        # only consider MCS that contain the anchor.
        if mol_anchor:
            # only consider MCS that is larger than allowance * n atoms.
            if len(mol_indices) >= self._allowance * len(fragment.GetAtoms()):
                self._mcs_found = True
                return True

        return False
