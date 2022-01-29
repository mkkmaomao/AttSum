# AttSum

Code for paper(under review) "AttSum: A Deep Attention-Based Summarization Model for Generating Bug Report Titles"

AttSum is an Encoder-Decoder model incorporating with the copy mechanism for bug report title generation.

1. Obtain the preprocessed dataset from the google drive: https://drive.google.com/drive/folders/1Q0zxfFkRP7qjnuRe_3dsRxDN7QywDx-h?usp=sharing

2. new_data folder contains 1) json_counting.py for data statistics; 2) and a folder including 50 bug reports with low-quality titles.

3. models folder contains the code for AttSum implementation, we do not upload the codes for reproducing baselines since they can be founded through the corresponding articles.

4. data_tools folder includes two .py files for data processing and automatic evaluation.

5. main.py is for running the program, and mconfig.py is for parameter settings.

