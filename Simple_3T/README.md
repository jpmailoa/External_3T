# Simple 3T

This sub repository allows the usage of 3T structure optimization / energy minimization algorithm in a significantly simplified manner, and currently it is written for single-molecule optimization application as a proof of concept. This sub repository uses the same algorithm / architecture as the main repository, but is performed on ASE Atoms object (single organic molecule is currently expected due to the off-the-shelf force field parametrization constraints). This simple 3T optimizer is compatible with an external ASE calculator (NWChem ASE calculator example is demonstrated).

## Quick Start

### Install Dependencies
Install all the dependencies and libraries required by the main repository.

### Ensure Correct `python` Command
Follow the instruction in the main repository.

### Correctly Configure Your Python ASE Calculator
Make sure that the ab-initio software you'd like to couple with 3T calculator is configured correctly with ASE.

### Test Run
At this point, you are ready to run a test example. We have provided a test molecule `Raw_12951679.xyz` which is basically the raw `.xyz` file of TE4PBA cation downloaded from PubChem. Run the following command for testing correct functionalities (just using off-the-shelf force field, without using the coupled ASE calculator):
```
python example.py
```

The files such as `3T.xyz` optimization trajectory and `3T_outE.txt` energy minimization curve should be generated if the environment is setup correctly. If they are, feel free to modify `example.py` for a real run using your preferred ab-initio ASE calculator.

### Issues
This sub repository is only made for demonstration purposes as a proof-of-concept demonstrating the reduction in required minimization steps when multi-scale gradient 3T optimizer is utilized. However, it will be slow because currently the 3T optimizer does not utilize the i-PI socket protocol. This means currently the ab-initio software is reboot on each 3T molecule optimization step (significant computation cost overhead).

## Contact

Questions about this repository may be addressed to Jonathan Mailoa ( jpmailoa [AT] alum [DOT] mit [DOT] edu ).
