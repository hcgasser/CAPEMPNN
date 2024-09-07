import os
import numpy as np
import pandas as pd
import pickle

from kit.log import log_info
from kit.bioinf.populations import NMDP
from kit.bioinf.pwm import count_amino_acid_occurrences
from kit.bioinf.immuno.utils import get_immuno_setup_hash


def get_agg_mhc_1_alleles(agg_specification, populations, df_haplotype_feqs):
    try:
        min_coverage = float(agg_specification)

        # get the haplotypes that cover at least "min_coverage" of any sub-population
        _population_haplotypes = NMDP.get_mhc_1_haplotypes_min_coverage(
            min_coverage, populations, df_haplotype_feqs
        )

        # get the alleles that are present in those haplotypes
        _mhc_1_alleles = NMDP.mhc_1_haplotypes_to_alleles(
            [haplotype 
                for pop_haplotypes in list(_population_haplotypes.values()) 
                for haplotype in pop_haplotypes
            ]
        )
        agg_name = f"min_coverage_{min_coverage*100:.0f}pc"
    except ValueError:
        # single individual
        _population_haplotypes = None
        _mhc_1_alleles = sorted([_allele for _allele in agg_specification.split('+')])
        _mhc_1_hash = get_immuno_setup_hash({"mhc_1": _mhc_1_alleles})
        agg_name = f"p_{_mhc_1_hash}"
        _mhc_1_alleles = [_allele.removeprefix('HLA-') for _allele in _mhc_1_alleles]
        
    return _mhc_1_alleles, _population_haplotypes, agg_name


def generate_agg_pwm_mhc_1(agg_specification, populations, df_haplotype_feqs, predictor_mhc_1_pwm):
    log_info(f"Agg Specification: {agg_specification}")
    folder = predictor_mhc_1_pwm.data_dir_path

    log_info("    get agg_mhc_1 alleles")
    mhc_1_alleles, population_haplotypes, agg_name = get_agg_mhc_1_alleles(
        agg_specification, 
        populations, 
        df_haplotype_feqs
    )

    if population_haplotypes:
        pickle.dump(
            population_haplotypes, 
            open(os.path.join(folder, "definitions", f"{agg_name}_population_haplotypes.pkl"), "wb")
        )
        for population, population_haplotypes in population_haplotypes.items():
            log_info(f"    \tPopulation: {population} - {len(population_haplotypes)} haplotypes")

    log_info("    load_agg ranks")
    predictor_mhc_1_pwm.load_agg_ranks(agg_name, mhc_1_alleles)

    log_info("    generate PWMs")
    # generate position weight matrices
    peptides_presented = predictor_mhc_1_pwm.get_presented_any(agg_name)
    counts = count_amino_acid_occurrences(peptides_presented)
    predictor_mhc_1_pwm.calc_PWMs(agg_name, counts)
    predictor_mhc_1_pwm.save_PWMs(agg_name)


    log_info("    calculate percentile statistics")
    percentile_statistics = predictor_mhc_1_pwm.calc_percentile_statistics(agg_name)    
    predictor_mhc_1_pwm.save_percentile_statistics(agg_name, percentile_statistics)
