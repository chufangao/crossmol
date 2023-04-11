# crossmol

Download grover and add it to this directory: https://github.com/tencent-ailab/grover/tree/main/grover

crossmol_dataset.py = functions to load the chebi20 dataset for use in main model training
ddi_data.py = functions to load ablation experiment data
link_pred.py = abalation experiments for molecule to molecule link prediction
main.py = main model, as well as training
main_no_ca.py = main model without cross attention, as well as training
test.ipynb = main evaluation code