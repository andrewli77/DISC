# Interpretable Sequence Classification via Discrete Optimization 

This repository contains the code for DISC and the other baselines from the paper "Interpretable Sequence Classification via Discrete Optimization" by Shvo, Li, Toro Icarte, and McIlraith. Running DISC requires Gurobi Optimizer (available at https://www.gurobi.com) and an active license (academic licenses can be obtained for free). DISC is implemented in the files {DFA_utils_tree_minerror, read_traces, run_tree_model, tree_minerror, tree_utils}.py and is run from run_tree_model.py.

 For DISC, LSTM (run from lstm.py), and HMM (run from supervised_hmm.py), the file GLOBAL_VARS.py contains important flags. `VALIDATION` determines whether part of the training data will be used as a validation set in order to choose hyperparameters (true for the majority of experiments in the paper). `TREE_MINERROR`, if true, uses DISC when run_tree_model.py is run, and otherwise runs DFA-FT, another baseline in the experiments. `MULTILABEL`, if true, assumes there can be multiple labels associated with each observation trace and should be false unless you're running the multi-label classification experiment. 

 The files by default use the settings specified for most of the experiments in the paper, but note some exceptions (e.g. MIT-AR) for which you will have to go into the file and modify the settings manually. For example, for MIT-AR the possible transition penalty values (run_tree_model.py:151) should be changed to {3, 5.47, 10}, q_max to 5 (run_tree_model.py:296 or from the command-line), and `VALIDATION` to False (in GLOBAL_VARS.py).

 The random training/testing splits of the datasets, as well as recorded results, can be found in the traces/ folder. 