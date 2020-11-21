# Readme.md


Getting started
===========

To install the 'ProtPred' environment from `environment.yml` run:
> conda env create -f environment.yml
> conda activate ProtPred

Folder structure
----------------

The following Folder structure is expected:

>├── data
>│   ├── annotations
>│   │   └── merged_anno.txt
>│   ├── ec_vs_NOec_pide100_c50.h5
>│   ├── lookupSets
>│   │   ├── ec_vs_NOec_pide100_c50.fasta
>│   │   ├── ec_vs_NOec_pide20_c50_test.fasta
>│   │   ├── ec_vs_NOec_pide20_c50_train.fasta
>│   │   ├── ec_vs_NOec_pide20_c50_val.fasta
>│   │   ├── ec_vs_NOec_pide30_c50.fasta
>│   │   ├── ec_vs_NOec_pide40_c50.fasta
>│   │   ├── ec_vs_NOec_pide50_c50.fasta
>│   │   ├── ec_vs_NOec_pide60_c50.fasta
>│   │   ├── ec_vs_NOec_pide70_c50.fasta
>│   │   ├── ec_vs_NOec_pide80_c50.fasta
>│   │   └── ec_vs_NOec_pide90_c50.fasta
>│   └── nonRed_dataset.tar.gz
>├── data_stats.py
>├── environment.yml
>├── h5opener.py
>├── readme.md
>├── Simple_data_exploration.ipynb
>├── Untitled.ipynb
>└── utils.py


