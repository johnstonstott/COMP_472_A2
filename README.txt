Assignment 2
Written by Johnston Stott (40059176)
For COMP 472 Section ABIX – Summer 2020

Required Packages
-----------------
- matplotlib
- numpy
- scikit-learn

Required Version
----------------
- Python 3

Instructions
------------
1. Install the required packages listed above.
2. In the command line, navigate to the directory containing the main.py file. The dataset files MUST be in the same
directory as main.py, otherwise the program will not work.
3. Begin the program by running the main.py file with Python 3. I used the following command:

python3 main.py

4. When prompted, enter the file name for the training set, with its file extension. The file MUST be in the same
directory as main.py.
5. When prompted, enter the file name for the testing set, with its file extension. This can be the same as the training
set. The file MUST be in the same directory as main.py
6. Read the messages in the command line window to see which step the program is at. The program will perform all tasks
and then terminate.

IMPORTANT: The program will generate output files ONLY if these files DO NOT already exist. The names of the output
files are:

- model-2018.txt
- vocabulary.txt
- baseline-result.txt
- stopword-model.txt
- stopword-result.txt
- wordlength-model.txt
- wordlength-result.txt

Since these files are included in my submission, by default the program WILL NOT create new output files. If you would
like to see the program generate new output files yourself, you must either delete the existing output files, rename
the existing output files to something other than the above list, or move the existing output files to another
directory, not in the same location as main.py.

Example Program Execution
-------------------------

Below, I have given an example run-through of the program and the format of the accepted input in case you are getting
it incorrect.

Enter the file name for the data set (e.g.: hns_2018_2019.csv):
hns_2018_2019.csv <Enter>

Enter the file name for the testing set (e.g.: hns_2018_2019.csv):
hns_2018_2019.csv <Enter>

<The program will perform tasks 1, 2, 3.1, 3.2, and generate the respective output files, ONLY if they are not already
present. For task 3.3, the program will perform the first classifier and display a graph. When you are done viewing the
graph, you may close it and the program will resume with the second classifier. Another graph will be displayed when
calculations are finished. When you are done viewing it, you can close the graph and the program will terminate.>
