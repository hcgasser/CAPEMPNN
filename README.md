<center><img src="logo.jpg" alt="Cape Logo" width="200" height="200"><br><br><br></center>

This is the temporary repository for CAPE-MPNN during the review process. The code will eventually be included into the larger CAPE repository (https://github.com/hcgasser/CAPE).


# Controlled Amplitude of Present Epitopes (CAPE)

Protein therapeutics already have an arsenal of applications that include disrupting protein interactions, acting as potent vaccines, and replacing genetically deficient proteins. 
Therapeutics must avoid triggering unwanted immune-responses towards the therapeutic protein or viral vector proteins. 

Therefore, computational methods modifying proteins' immunogenicity are needed. 


## CAPE-MPNN

We apply DPO on **ProteinMPNN** to tune it to incorporate less MHC Class I epitopes into its generations.
The new **ProteinMPNN** weights generated this way, we call **CAPE-MPNN**.


### Installation

#### General Requirements
- The programs in this repository require a Linux machine
- Colabfold needs to be installed (https://github.com/sokrypton/ColabFold)

#### Clone the CAPE repository
- Clone repository to local machine: ``git clone https://github.com/hcgasser/CAPE_MPNN.git``
- Make environmental variable pointing to the repo folder: ``export CAPE=<path of repo folder>``
- add ``CAPE`` to ``.bashrc`` file: ``echo "export CAPE=${CAPE}" >> ~/.bashrc``


#### Clone the original ProteinMPNN repository
- make external repos directory: ``mkdir -p $CAPE/external/repos``
- change to this directory: ``cd $CAPE/external/repos``
- clone the repository: ``git clone https://github.com/dauparas/ProteinMPNN.git``


#### Download the original training data from the ProteinMPNN paper
- make external data directory: ``mkdir -p $CAPE/external/data/ProteinMPNN``
- change to this directory: ``cd $CAPE/external/data/ProteinMPNN``
- download the multi-chain training data (16.5 GB, PDB biounits, 2021 August 2): ``wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz``
- unpack: ``tar -xzf pdb_2021aug02.tar.gz; rm pdb_2021aug02.tar.gz``



### Run

At first we run a script to setup the environment
```shell
cd $CAPE
. ./tools/set_ENV.sh
```

Then we set the MHC-I alleles to deimmunize against
``export MHC_Is="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"``


#### Generate the MHC Class I position weight matrices
``MHC-I_rank_peptides.py --output ${CAPE}/data/input/immuno/mhc_1/Mhc1PredictorPwm --alleles ${MHC_Is} --tasks rank+pwm+agg --peptides_per_length 1000000``



### DPO hyperparameter search

```shell 
export HOSTNAME='workstation'
cape-mpnn.py --hyp ${PF}/configs/CAPE-MPNN/hyp/hyp_b69bb1.yaml --hyp_n 20
```

### Eval

The evaluation jupyter notebook can be found in ``CAPE-Eval/cape-eval_mpnn.ipynb``