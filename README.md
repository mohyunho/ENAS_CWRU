# ENAS_CWRU
Evolutionary Neural Architecture Search for Fault Diagnosis on CWRU. <br/>
- Fault diagnosis (multi-class classification). <br/>
- Bearing Data [CWRU](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website) <br/>
- Data description in this [article](https://www.sciencedirect.com/science/article/pii/S1474034616301148) <br/> 
- Recent survey papers in this field [link](https://ieeexplore.ieee.org/abstract/document/8988271) [link](https://ieeexplore.ieee.org/document/9078761)
- Run neural architecture search for MLPs (TODO) <br/>


## Prerequisites
The library has the following dependencies:
```bash
pip3 install -r requirements.txt
```
-pandas <br/>
-numpy <br/>
-matplotlib <br/>
-seaborn <br/>
-sklearn <br/>
-pyts <br/>
-scikit-learn <br/>
-tensorflow-gpu <br/>
-pathos <br/>

## Descriptions
- fault_diagnostics.py: launcher for the experiments.
  - cwru_custom.py: download and create data 
  - fd_network.py:
  - ea.py


## Run
Please launch the experiments by 
```bash
python3 fault_diagnostics.py -i datafile -l seq_length 
```

&ndash;  i : input datafile (12 for 12k Drive End Bearing Fault Data & 48 for 48k Drive End Bearing Fault Data) <br/>
&ndash;  l : sequence length of time series <br/>
--cross : default='no', help='10-fold cross validation 'yes' or 'no'' <br/>
--dim_method : default=non, help='dim.reduction methods, 'non', 'sfa' or 'pca'' <br/>
--n_comp : default=100, help='number of dim reduction components' <br/>

You can check all the other arguments and their details by
```bash
python3 fault_diagnostics.py -h
```

- Data files are automatically downloaded at the first time
- 9/10 of data used for train and 1/10 for test (There is no original training & test split on CWRU). 
- the number of output classes is 10 (one normal and 9 different faults).
- If cross validation is validated, the network evaluated 10 times and the results(accuracy) are averaged.
- please check this [article](https://www.sciencedirect.com/science/article/pii/S1474034616301148) for the data description 

For example,
```bash
python3 fault_diagnostics.py -i 48 -l 400 --dim_method non --epochs 1000 --cross no --plotting no
```

Output print
```bash
accuracies:  [0.9180999180999181, 0.8, 0.8578431372549019]
avg accuracy:  0.8586476851182733
```
