#!/bin/bash
#SBATCH --mem=20gb
#SBATCH -c10
#SBATCH --time=1-0
#SBATCH --error=array_error_log_job%A_id%a.txt
#SBATCH --output=array_log_job%A_id%a.txt
#SBATCH --job-name=clustering_test
source ../DynamicRepresentations/new_env/bin/activate
python3 Clustering.py --visualize_clusters --visualize_abs_clusters --num_of_points 50000 --embedding_batch_size 10000 --eps 0 --min_samples 2 --num_clusters 100 --loose_clustering_type kmeans --loose_num_clusters 100 --clustering_type agglomerative --distance_threshold 0.2 --entailment_threshold 0.5 --entail_prefix "I want" --nli_model "v3_large" --purpose_type GPT3