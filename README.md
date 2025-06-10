# MDP-GRL: Multi-disease Prediction by Graph-enabled Representation Learning
---

Source code for MDP-GRL

---

##  Requirements
- python 3.7
- pytorch == 1.13.1
- numpy == 1.21.5
- pandas == 1.3.5
- scikit-learn == 1.0.2
- tqdm == 4.64.1
---
##   Run the Codes
To train on a custom medical dataset, please place your data in the  `datasets` folder.

For example, this model is trained using the MIMIC dataset. We place the `MIMIC` folder under the datasets directory. You can run the following command to start the training:
```bash
python main.py --data_name mimic  
```

After training is completed, the model parameter files will be stored in the `trained_model/MDP_GRL/mimic`  directory. The best model will be saved as `best_model.pth`  in the same directory.

##  Open Medical Knowledge Graph Dataset
The dataset used in this model is a custom-built knowledge graph based on the publicly available MIMIC dataset.

The custom Medical Knowledge Graph dataset is located at `datasets/mimic/mimic_final_kg.txt`.  
The training dataset can be found in `datasets/mimic/train1_patient.txt`.  
The testing dataset can be found in `datasets/mimic/test1_patient.txt`.