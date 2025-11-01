# MUSE
Code and data for the paper "MUSE: Mining Unexpected Solutions Engine" \n
Paper: https://arxiv.org/abs/2509.05072. \n
For questions, please contact nir.sweed@mail.huji.ac.il. 

This repository contains:
1. Graph files - the files for a Functional Concept Graph constructed from 500k patents extracted from the US patent dataset. Released for future work.
2. Grpah code - code and data used for creating the FCG.
3. Experiment code - code used for the user-study described in the paper.


## Graph Files



## Graph_code
The graph creation process is divided into a few separate steps.
  1. Getting graph nodes with clustering (Clustering.py). An example for running this step is given in clustering_batch.sh
  2. Getting graph edges via entailment (ClusterEntailmentInBatch.py). An example for running this step is given in entail_batch.sh. This step is done separately for each loose cluster. Note this code is optimized for SLURM and is meant to run the entailment process concurrently on many SLURM machines. Setting total_jobs = 1 will cause this to run on a single machine. 
  3. Finding candidate nodes (CreateInterconnectingCands.py). These nodes are used for connecting between loose clusters (CreateInterconnections.py) and enhancing graph connectivity with LLM and Verb-based abstractions.  Example for running both stages is found in interconnections_batch.sh
  4. 
