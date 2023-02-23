# PhysioNet 2012 Challenge dataset #

### Raw data###
* /rawdata/
  * Raw data: set-a, set-b, set-c, Outcomes-a.txt, Outcomes-b.txt, Outcomes-c.txt
  * Data source: https://www.physionet.org/content/challenge-2012/1.0.0/

### Data parsing (process_scripts) ###
Run following scripts in turn:
* /process_scripts/
  * ParseData.py : generate arr\_outcomes.npy, ts\_params.npy, static\_params.npy, and P\_list.npy
  * IrregularSampling.py: generate: extended\_static\_params.npy, PTdict\_list.npy
  * Generate\_splitID.py: generate phy12\_splitX.npy where X range from 1 to 5. Only contains the IDs. Train/val/test =  9:1:1

Note: PTdict\_list.npy and arr\_outcomes.npy are the most important files.


### Processed data ###:
* /processed_data/
  * PTdict_list.npy
  * arr_outcomes.npy  
  * ts_params.npy
  * extended_static_params.npy
  
  * static_params.npy
  * P_list.npy
* /splits/
  * phy12\_splitsX.npy  where X range from 1 to 5. In splits folder: there are 5 npy files, each contains three array representing idx\_train,idx\_val,and idx\_test.
  
