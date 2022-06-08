# python infer.py --dataDir "data/lipread_mp4_demo_processed_1000" --batchSize 64 --targetDir "data/lipread_mp4_demo_processed_1000/landmarks" --numWorkers 8 --glob "*/*/*/*/*.jpg" --demo_dir "lipread_mp4_demo_processed_1000_demo" --demo

# python infer.py --dataDir "data/lipread_mp4_demo_processed" --batchSize 128 --targetDir "data/lipread_mp4_demo_processed/landmarks" --numWorkers 8 --glob "*/*/*/*/*.jpg" --demo_dir "lipread_mp4_demo_processed_demo" --demo --demo_count 100

# python infer.py --dataDir "data/lipread_mp4_processed" --batchSize 128 --targetDir "data/lipread_mp4_processed/landmarks" --numWorkers 8 --glob "*/*/*/*/*.jpg" --demo_dir "lipread_mp4_processed_demo" --demo --demo_count 100

# CUDA_VISIBLE_DEVICES=1 python infer.py --dataDir "/home/arjun.ashok/files/TwoStageDubbing/inference/24-05-2022-11:55:05" --batchSize 128 --targetDir "/home/arjun.ashok/files/TwoStageDubbing/inference/24-05-2022-11:55:05/landmarks" --numWorkers 8 --glob "*/*.jpg" --demo_dir "inference-demo-EARLY_00001"