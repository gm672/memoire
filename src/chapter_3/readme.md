# Chapter 3
These scripts have been used in the third chapter. The files all the files except dl.py et report.py have been taken from the Alearbres project : https://github.com/marinecourtin/Alearbres. All the modifications I made to theses scripts have been notified at the beginning of each file.

## Usage
To generate a csv file containing all the measures from a connlu file, use the script report.py. The script takes two positional argument : the input file and the output file.

Example command :

> python report.py fro_srcmf-ud-test.conllu report_old_french_test.csv

The output CSV file is formated with one sentence per line with  :

> [length, actual dependency length, random DL , minimum DL, omega, gamma, MDD]

as collumns.