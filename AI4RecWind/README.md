# AI4RecWind (Artificial Intelligence for the Reconstruction of Wind) by Climatoc-Lab

Software to train/evaluate models to reconstruct missing values in historical observations of windspeed from AEMET's network based on a U-Net with partial convolutions (PCNN's model).

## Versions of the PCNN's model

### PCNN version by Lihong Zhou: PCNN4RecWind_Zhou

Software created by Lihong Zhou and Haofeng Liu, based on the code developed and used for HadCruT's reconstruction in Kadow et al. (2020). Applied for the reconstruction of windmaps based on the HadISD's windspeed dataset at global scale, Zhou et al. (2022) (https://doi.org/10.1016/j.scib.2022.09.022).

#### Use in Climatoc-Lab:
This version of the PCNN neural network has been used by Climatoc-Lab to obtain the first results of the reconstruction of the daily windspeed maps using the historical wind observations from AEMET's weather stations network.

### New PCNN version: : PCNN4RecWind_Climatoc

Software created by the Data Anlysis Group, leaded by Christopher Kadow, at Deutsches Klimarechnunszentrum (DKRZ). This software is based on the code developed and used for HadCruT's reconstruction in Kadow et al. (2020) (https://doi.org/10.1038/s41561-020-0582-5).

#### License

`CRAI` is licensed under the terms of the BSD 3-Clause license.

#### Contributions

`CRAI` is maintained by the Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum).
- Previous contributing authors: Naoto Inoue, Christopher Kadow, Stephan Seitz
- Current contributing authors: Johannes Meuer, Maximilian Witte, Étienne Plésiat.

#### Use in Climatoc-Lab:
This version of the PCNN neural network is the new one used by Climatoc-lab to obtain the new reconstructions of the daily windspeed maps. We have chosen it as it includes new implementations (LSTM, GRU, Attention mechanism) to be implemented in the PCNN's model.



