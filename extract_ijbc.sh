#!/bin/bash -x
CUDA_VISIBLE_DEVICES=0 python -m evaluation.extract_embeddings --meta_path /s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJBC_backup.npz\
    --model_path pretrained/PFE_sphere64_casia_am \
    --input_path IJBC_align_96x112/loose_crop \
    --output_path ijbc_embs_pfe_sphere64_casia_am.npy
CUDA_VISIBLE_DEVICES=0 python -m evaluation.extract_embeddings --meta_path /s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJBC_backup.npz\
    --model_path pretrained/PFE_sphere64_msarcface_am \
    --input_path IJBC_align_96x112/loose_crop \
    --output_path ijbc_embs_pfe_sphere64_msarcface_am.npy
