# A Neural Network Pitch Prediction Model

This repository contains files and data pertaining to multiple versions of Neural Network models designed to predict what the next pitch will be in a given at-bat. The file names and their purposes are listed below:

**_Important Note_**: some of the index slicing may need to be changed in the files below depending on whether or not your dataset includes indices. These were written to work with a dataset that included indices in the first column.

**R Scripts**
- *PSM_Data_Scraper.R* - Uses the *baseballr* package to scrape and save Statcast pitch-by-pitch data
- *PSM_Data_Builder.R* - Processes the raw data into various input formats for the python scripts below

**Python Scripts**
- *PSM_DNN_OHE.py* - Builds a feed-forward network with one-hot-encoded categorical features
- *PSM_DNN_EMB.py* - Builds a feed-forward network with embedded categorical features
- *PSM_RNN_OHE.py* - Builds an RNN with one-hot-encoded sequences and context vectors
- *PSM_RNN_MIX.py* - Builds an RNN with embedded sequences and one-hot-encoded context vectors
- *PSM_RNN_EMB.py* - Builds an RNN with embedded sequences and context vectors

**Data Files**
- *DNN_EMB_results.csv* - Summary of prediction results for 1133 pitcher seasons across 2018-2023 (2020 excluded)
- *Miley_Data.csv* - A sample data set containing every pitch Wade Miley threw in 2021. For use with *PSM_DNN_EMB.py*
