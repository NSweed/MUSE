#!/bin/bash
#SBATCH --mem=20gb
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH -c10
#SBATCH --time=4-0
#SBATCH --error=array_error_log_job%A_id%a.txt
#SBATCH --output=array_log_job%A_id%a.txt
#SBATCH --job-name=500k_entailment_test
#SBATCH --array=104-104%50
source ../DynamicRepresentations/new_env/bin/activate
python3 ClusterEntailmentInBatch.py --job_id $SLURM_ARRAY_TASK_ID --visualize_clusters --visualize_abs_clusters --num_of_points 500000 --embedding_batch_size 50000 --eps 0 --min_samples 2 --num_clusters 5 --loose_clustering_type kmeans --loose_num_clusters 1000 --clustering_type agglomerative --distance_threshold 0.2 --entailment_threshold 0.5 --entail_prefix "I want" --nli_model "v3_large" --total_jobs 500