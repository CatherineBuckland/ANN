# ANN
Two artificial neural networks to define the relationship between disturbance factors and sediment deposition in dryland environments

** MODEL OVERVIEW **

Two ANNs are presented here to explore the relationship between sediment deposition in aeolian environments and the sensitivity to external factors. 

The first ANN (ANN1) is designed to identify the existing relationship between tree ring growth, land use pressure and wildfire with the likelihood of identifying an episode of sediment deposition in dryland aeolian environments. 
Six separate experiment sites are trained alongside each other in ANN1 to maximise the training dataset of the model. 

The second ANN (ANN2) is used to generate new predicted tree ring growth datasets based on hypothetical climate futures (i.e. under different temperature and precipitation regimes). The new tree ring growth data can then be added to new levelf of land use pressure and wildfire occurrence to predict future landscape sensivitiy. 

Both ANNs require paired input-target datasets for initial model training with predicted target datasets produced using new input series. 

The model is implemented Matlab.

** RUNNING THE MODEL **

The main ANN code files are 'ANN1_code.m' and 'ANN2_code.2' 
'SiteA....-F.csv' files represent ANN1 training datasets. A minimum of one site is needed for training purposes and the user can alter the number of training sites included in the training and testing of the ANN. 
'Climate_training.csv' is the training dataset for ANN2. 
'Climate1.csv' and 'Climate2.csv' represent two new hypothetical climate training datasets. 
Scenario.csv files refer to new hypothetical disturbance scenarios to be used in ANN1 to generate new landscape sensitivity scores. LG refers to low grazing pressure, MG medium grazing pressure, and HG high grazing pressure. 

For further details RE cross-validation, model performance and scenario testing please contact catherine.buckland@ouce.ox.ac.uk
