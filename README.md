The given code is designed to perform intelligent sampling of molecular structures and corresponding X-ray absorption near-edge structure (XANES) spectra data. 
It allows users to specify directories containing molecular structure files in XYZ format and XANES spectra data, and select the number of samples to extract based on principal
component analysis (PCA). The script leverages the `WACSF` descriptor from `xanesnet` to transform XYZ files into numerical descriptors, which are then used for sampling. 
The intelligent sampling is controlled via the command line arguments, which specify whether the sampling should be performed on molecular structures (`xyz`) or spectra data (`xanes`).


When the user opts for molecular structure sampling, the script reads molecular XYZ files from the specified directory, applies the `WACSF` descriptor to extract features, 
and then applies PCA to reduce the dimensionality of the feature space. The sampling process selects molecular structures that are maximally dissimilar to each other in 
the reduced feature space, ensuring that the chosen samples are representative of the entire dataset. The script outputs the sampled XYZ files into separate directories based 
on the number of samples specified by the user. Additionally, the corresponding XANES spectra files for the selected structures are also copied into the output directories.

For XANES spectra sampling, the script reads the spectra data, applies PCA to reduce the dimensionality, and performs intelligent sampling in a similar fashion. This ensures
that the selected spectra are diverse and representative of the entire dataset. The script creates directories for the chosen spectra files and their corresponding XYZ files,
which are copied based on the intelligent sampling process.

In both modes (`xyz` and `xanes`), the script iterates over the sampling steps specified by the user, creating output directories for each step and copying the selected files. 
The intelligent sampling approach ensures that the chosen molecular structures or spectra are diverse and spread out in the feature space, providing a meaningful subset of data
for further analysis.


For more details please refer https://github.com/NewcastleRSE/xray-spectroscopy-ml
