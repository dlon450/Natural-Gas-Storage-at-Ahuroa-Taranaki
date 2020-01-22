# Gas Storage Project

## Description
This project studies the pressure changes in the Ahuroa Gas Storage Site owned by FlexGas. By using calibration and numerical integration, it computes the gas leakage in the reservoir - both current and future.

This git repository contains the following files:
- main.py: this contains all the modelling and plotting functions. Running this file will produce all the relevant plots one after the other.
- gs_functions.py: this file contains the key functions used in main.py for calibration and integration.
- test_gs.py: this file contains the benchmarking code and unit tests to verify the accuracy of the numerical integration implemented in solve_ode in gs_functions.

In addition to this, there is a folder called Data containing the files gs_mass.txt and gs_pressure.txt which contain the monthly reservoir mass flow rate and quarterly pressure data, respectively.

## Usage

To use this repository, clone it and run the main.py file. Ensure that the files are in the same folder.

## Framework
Built with Visual Studio Code, using Python Programming Language.

## Support
For any issues or concerns, contact us:  
jmis297@aucklanduni.ac.nz  
jtri372@aucklanduni.ac.nz  
jim853@aucklanduni.ac.nz  
dlon450@aucklanduni.ac.nz

## Authors and Acknowledgement
James Missen, Jahnvi Trivedi, Joon Im, Derek Long.  
Special thanks to Dr. David Dempsey and 263 Lab Tutors/Technicians.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.  

Please make sure to update tests as appropriate.