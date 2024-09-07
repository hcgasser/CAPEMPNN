<center><img src="logo.jpg" alt="Cape Logo" width="200" height="200"><br><br><br></center>

This is the temporary repository for CAPE-MPNN during the review process. The code will eventually be included into the larger CAPE repository (https://github.com/hcgasser/CAPE).


# Controlled Amplitude of Present Epitopes (CAPE)

Protein therapeutics already have an arsenal of applications that include disrupting protein interactions, acting as potent vaccines, and replacing genetically deficient proteins. 
Therapeutics must avoid triggering unwanted immune-responses towards the therapeutic protein or viral vector proteins. 

Therefore, computational methods modifying proteins' immunogenicity are needed. 


## CAPE-MPNN

We apply DPO on **ProteinMPNN** to tune it to incorporate less MHC Class I epitopes into its generations.
The new **ProteinMPNN** weights generated this way, we call **CAPE-MPNN**.

## DPO hyperparameter search

```shell 
cd $PF
. ./tools/set_ENV.sh
export HOSTNAME='workstation'
cape-mpnn.py --hyp ${PF}/configs/CAPE-MPNN/hyp/hyp_b69bb1.yaml --hyp_n 20
```

## Eval

The evaluation jupyter notebook can be found in ``CAPE-Eval/cape-eval_mpnn.ipynb``