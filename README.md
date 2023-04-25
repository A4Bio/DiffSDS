
# Dataset preperation
1. download dataset
    ```
    cd data
    wget -P cath ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz

    cd cath
    tar -xzf cath-dataset-nonredundant-S40.pdb.tgz
    ```
2. split data
    ```
    cd data
    python datasplit.py
    ```

# Train the model
- Debug: 

    Firstly, enter the following commands:
    ```
    CUDA_VISIBLE_DEVICES="0" python -m debugpy --listen 5698 --wait-for-client -m torch.distributed.launch --nproc_per_node 1 main.py
    ```
    then, use VSCode to debug the `main.py`.

- Run:

    ```
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --ex_name DiffSDS --method DiffSDS
    ```

- Sampling:
    ```
    sh sampling.sh
    ```

# Released assets