Installation instructions:

1. Install Tracking-Anything-with-DEVA according to the instructions here- https://github.com/hkchengrex/Tracking-Anything-with-DEVA/tree/main

2. Replace the included files in Tracking-Anything-with-DEVA/deva/inference and Tracking-Anything-with-DEVA/deva/inference/data with the provided code. Note that the other files present do not need to be modified.

3. Install the python libraries in packages.txt

Use instructions:

To create data:

1. Set the values in config.py to your local directory structure

2. Apply Segment Anything to the frames of the desired video sequence. We used the prompts \['paper', 'cardboard', 'metal', 'plastic', 'hand'\] but this is not necessary. 

3. Place the unmodified, numbered frames of the input video sequence in data/raw/\[sequence\_name\] and the output of SAM in data/masks/\[sequence\_name\]

4. In config.py, set IMGS to a list of each \[sequence\_name\] you wish to process

5. Run sorting\_DEVA.py. 

To convert files to datasets:

1. Set the directories in coco\_to\_tfrecord.ipynb for your local system.

2. Run the notebook.

To retrain models:

1. Set the directories in modelRetraining.py for your local system and dataset (see below for datasets used in the paper)

2. Run massEval.py

To apply models:

1. Set the directories in massEval.py for your local system and checkpoint

2. Run massEval.py

To evaluate data:

1. In predictionEvaluation.ipynb, set "targets" in cell 3, line 20 to a list of the output label files you wish to examine. By default, these are found in data/5\_5/Labels

2. In predictionEvaluation.ipynb, set "testSource" in cell 3, line 10 to the ground truth labels you wish to compare against.

3. Run predictionEvaluation.ipynb

Dataset organization:
sequences are labeled by difficulty and index- easy\_1, easy\_2, etc...

easy\_hand, hard\_hand, and med\_1\_hand are validation sequences that have been hand annotated using cvat.ai. med\_1\_hand contains 30 hand annotated frames from the med\_1 sequence, while the other two contain hand annotated frames from sequences not used in training. med\_1\_hand was used to determine if the training data could produce any improvements, while easy\_hand and hard\_hand were used for the response to training tests.

output contains the raw output of DEVA on the med\_1 sequence.

tf\_record\_groups:
med\_XX.X and hard\_XX.X are the training sets used for the training response study. The \_XX.X is the percentage of footage that was removed from the full sequence.

med\_1 is the training set produced by our method. med\_1\_sota and med\_1\_weak\_supervision are the training sets produced by DEVA and weak supervision, respectively.

