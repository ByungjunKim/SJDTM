1. realdata_twtm_input.pkl
This file has the following data: 
(1) the leading corpus and lagged corpus
(2) the time slice for leading and lagged corpus
(3) the setup of number of shared, lead-specific and lag-specific topics
(4) the setup of the maximum lag period
(5) the id2word, which represents the vocabulary
(6) the initilization of topic-word distributions.
More details can be found in real_data_code.py

2. real_data_code.py
This is the main program of SJDTM for analysing the real datasets.

3. DTMpart.py
This file is used for updating topic-word distributions in real_data_code.py

4. real_data_evalution_metrics.py
This file is used to compute the evalution_matrics to measure the performance of SJDTM.

Users can put the above files in one folder and then directly run real_data_code.py.