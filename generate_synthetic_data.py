"""
ProInX: Synthetic Data Generation for Proof-of-Concept Model
==============================================================

This script generates REALISTIC SYNTHETIC data mimicking NMR-derived local Kd 
measurements for protein-small molecule interactions.

IMPORTANT: This is SIMULATED data for methodological development only.
Real experimental validation is required before clinical application.

The synthetic data is based on:
1. Real HisJ NMR data patterns (Kd range: 7-200 mM)
2. Known physicochemical principles of AA-protein interactions
3. Published literature on patch-based binding models

Author: ProInX Project/Jonas
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os

# Seed for reproducibility
np.random.seed(42)

# =============================================================================
# AMINO ACID PROPERTIES (from literature)
# =============================================================================

AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'size': 89, 'aromatic': 0, 'name': 'Ala'},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'aromatic': 0, 'name': 'Arg'},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'size': 132, 'aromatic': 0, 'name': 'Asn'},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'size': 133, 'aromatic': 0, 'name': 'Asp'},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'size': 121, 'aromatic': 0, 'name': 'Cys'},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'size': 147, 'aromatic': 0, 'name': 'Glu'},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'size': 146, 'aromatic': 0, 'name': 'Gln'},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'aromatic': 0, 'name': 'Gly'},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'size': 155, 'aromatic': 1, 'name': 'His'},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'size': 131, 'aromatic': 0, 'name': 'Ile'},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'size': 131, 'aromatic': 0, 'name': 'Leu'},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'size': 146, 'aromatic': 0, 'name': 'Lys'},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'size': 149, 'aromatic': 0, 'name': 'Met'},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'size': 165, 'aromatic': 1, 'name': 'Phe'},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'aromatic': 0, 'name': 'Pro'},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'aromatic': 0, 'name': 'Ser'},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'size': 119, 'aromatic': 0, 'name': 'Thr'},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'size': 204, 'aromatic': 1, 'name': 'Trp'},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'size': 181, 'aromatic': 1, 'name': 'Tyr'},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'size': 117, 'aromatic': 0, 'name': 'Val'},
}

# Small molecule properties (stabilizers)
SM_PROPERTIES = {
    'proline': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'h_bond_donor': 1, 'h_bond_acceptor': 2},
    'glycine': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'h_bond_donor': 2, 'h_bond_acceptor': 2},
    'betaine': {'hydrophobicity': -0.5, 'charge': 0, 'size': 117, 'h_bond_donor': 0, 'h_bond_acceptor': 2},
    'serine': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'h_bond_donor': 3, 'h_bond_acceptor': 3},
    'arginine': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'h_bond_donor': 5, 'h_bond_acceptor': 2},
}

# =============================================================================
# PROTEIN DEFINITIONS (simplified sequences for simulation)
# =============================================================================

# Real proteins with known structures
PROTEINS = {
    'ubiquitin': {
        'pdb_id': '1UBQ',
        'length': 76,
        'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        'pI': 6.6,
    },
    'lysozyme': {
        'pdb_id': '1LYZ',
        'length': 129,
        'sequence': 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL',
        'pI': 11.4,
    },
    'BSA': {
        'pdb_id': '3V03',
        'length': 583,
        'sequence': 'DTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA',
        'pI': 4.7,
    },
    'barnase': {
        'pdb_id': '1A2P',
        'length': 110,
        'sequence': 'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
        'pI': 9.0,
    },
    'HisJ': {
        'pdb_id': '1HPB',
        'length': 238,
        'sequence': 'MKKLVLSLSLVLAFSSATAAFAAIPQNIRIGTDPTYAPFESKNSQGELVGFDIDLAKELCKRINTQCTFVENPLDALIPSLKAKKIDAIMSSLSITEKRQQEIAFTDKLYAADSRLVVAKNSDIQPTVESLKGKRVGVLQGTTQETFGNEHWAPKGIEIVSYQGQDNIYSDLTAGRIDAAFQDEVAASEGFLKQPAGKEYKVKVIQNQGKFVDYYADFVFAKKFATYGLNEGIFDLVGSGVKQDLVKNQP',
        'pI': 5.8,
    },
}

# =============================================================================
# SYNTHETIC DATA GENERATION FUNCTIONS
# =============================================================================

def calculate_patch_features(sequence: str, center_idx: int, window: int = 3) -> Dict:
    """
    Calculate features for a patch centered at center_idx.
    """
    start = max(0, center_idx - window)
    end = min(len(sequence), center_idx + window + 1)
    patch_seq = sequence[start:end]
    
    # Calculate patch properties
    hydrophobicity = np.mean([AA_PROPERTIES.get(aa, AA_PROPERTIES['G'])['hydrophobicity'] 
                             for aa in patch_seq])
    charge = sum([AA_PROPERTIES.get(aa, AA_PROPERTIES['G'])['charge'] for aa in patch_seq])
    aromatic_count = sum([AA_PROPERTIES.get(aa, AA_PROPERTIES['G'])['aromatic'] for aa in patch_seq])
    avg_size = np.mean([AA_PROPERTIES.get(aa, AA_PROPERTIES['G'])['size'] for aa in patch_seq])
    
    # Simulate solvent accessibility (higher for surface residues)
    # In reality this would come from structure
    sasa = 0.3 + 0.7 * np.random.beta(2, 2)  # Beta distribution centered around 0.5
    
    return {
        'hydrophobicity': hydrophobicity,
        'charge': charge,
        'aromatic_count': aromatic_count,
        'avg_size': avg_size,
        'sasa': sasa,
        'patch_length': len(patch_seq),
    }


def predict_local_kd(patch_features: Dict, sm_properties: Dict, protein_pI: float) -> float:
    """
    Predict local Kd based on physicochemical matching.
    
    This is a simplified model based on known principles:
    1. Hydrophobic patches bind proline/betaine better
    2. Charged patches bind oppositely charged SMs
    3. Aromatic patches show enhanced binding
    4. Surface accessibility matters
    """
    # Base Kd (in mM) - weak interactions
    base_kd = 50.0
    
    # Hydrophobic matching (proline likes hydrophobic patches)
    hydro_match = -0.5 * patch_features['hydrophobicity'] * sm_properties['hydrophobicity']
    
    # Charge interaction
    # Positive SM binds better to negative protein (and vice versa)
    charge_factor = -2.0 * patch_features['charge'] * sm_properties['charge']
    
    # Aromatic enhancement
    aromatic_bonus = -3.0 * patch_features['aromatic_count']
    
    # Surface accessibility (more accessible = better binding)
    sasa_factor = -10.0 * (patch_features['sasa'] - 0.5)
    
    # Size complementarity
    size_factor = -0.01 * abs(patch_features['avg_size'] - sm_properties['size'])
    
    # H-bond capability
    hbond_factor = -1.0 * (sm_properties['h_bond_donor'] + sm_properties['h_bond_acceptor']) / 10
    
    # Calculate log(Kd) with noise
    log_kd = np.log10(base_kd) + hydro_match + charge_factor + aromatic_bonus + sasa_factor + size_factor + hbond_factor
    
    # Add realistic noise (based on HisJ data variance)
    noise = np.random.normal(0, 0.15)
    log_kd += noise
    
    # Convert back to Kd and clamp to realistic range (5 mM - 500 mM)
    kd = 10 ** log_kd
    kd = np.clip(kd, 5, 500)
    
    return kd


def generate_protein_data(protein_name: str, protein_info: Dict, 
                          sm_name: str, sm_props: Dict,
                          n_interacting_residues: int = None) -> pd.DataFrame:
    """
    Generate synthetic NMR-like data for a protein-SM pair.
    """
    sequence = protein_info['sequence']
    length = protein_info['length']
    pI = protein_info['pI']
    
    # Determine number of interacting residues (typically 20-40% show interaction)
    if n_interacting_residues is None:
        n_interacting_residues = int(length * np.random.uniform(0.2, 0.4))
    
    # Select random surface residues (in reality, would be based on structure)
    all_residues = list(range(length))
    interacting_residues = np.random.choice(all_residues, size=n_interacting_residues, replace=False)
    
    data = []
    for res_idx in interacting_residues:
        res_aa = sequence[res_idx] if res_idx < len(sequence) else 'G'
        
        # Calculate patch features
        patch_features = calculate_patch_features(sequence, res_idx)
        
        # Predict Kd
        kd = predict_local_kd(patch_features, sm_props, pI)
        
        # Calculate error (proportional to Kd, as seen in real data)
        error = kd * np.random.uniform(0.1, 0.5)
        
        # Calculate CSP (inversely related to Kd - stronger binding = larger shift)
        csp_max = 0.1 + 3.0 * np.exp(-kd / 30)  # Higher CSP for lower Kd
        csp_max = np.clip(csp_max, 0.15, 4.0)
        
        data.append({
            'protein': protein_name,
            'pdb_id': protein_info['pdb_id'],
            'small_molecule': sm_name,
            'residue_id': f"{res_aa}{res_idx + 1}",
            'residue_num': res_idx + 1,
            'residue_type': res_aa,
            'kd_mM': round(kd, 4),
            'kd_error': round(error, 4),
            'csp_max': round(csp_max, 4),
            'hydrophobicity': round(patch_features['hydrophobicity'], 3),
            'charge': patch_features['charge'],
            'aromatic_count': patch_features['aromatic_count'],
            'sasa': round(patch_features['sasa'], 3),
        })
    
    return pd.DataFrame(data)


def generate_full_dataset() -> pd.DataFrame:
    """
    Generate complete synthetic dataset for multiple proteins and SMs.
    """
    all_data = []
    
    for protein_name, protein_info in PROTEINS.items():
        for sm_name, sm_props in SM_PROPERTIES.items():
            print(f"Generating data for {protein_name} + {sm_name}...")
            df = generate_protein_data(protein_name, protein_info, sm_name, sm_props)
            all_data.append(df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic NMR-like dataset...")
    print("=" * 60)
    
    dataset = generate_full_dataset()
    
    # Save to CSV
    dataset.to_csv("data/synthetic_local_kd_data.csv", index=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total data points: {len(dataset)}")
    print(f"Proteins: {dataset['protein'].nunique()}")
    print(f"Small molecules: {dataset['small_molecule'].nunique()}")
    print(f"Protein-SM pairs: {len(dataset.groupby(['protein', 'small_molecule']))}")
    print(f"\nKd range: {dataset['kd_mM'].min():.1f} - {dataset['kd_mM'].max():.1f} mM")
    print(f"Kd median: {dataset['kd_mM'].median():.1f} mM")
    
    print("\nData points per protein:")
    print(dataset.groupby('protein').size())
    
    print("\nData points per small molecule:")
    print(dataset.groupby('small_molecule').size())
    
    # Save summary
    summary = {
        'total_points': len(dataset),
        'proteins': list(dataset['protein'].unique()),
        'small_molecules': list(dataset['small_molecule'].unique()),
        'kd_range_mM': [float(dataset['kd_mM'].min()), float(dataset['kd_mM'].max())],
        'note': 'SYNTHETIC DATA for methodological development only'
    }
    
    with open("data/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data saved to data/synthetic_local_kd_data.csv")
    print("Summary saved to data/dataset_summary.json")
    print("=" * 60)

