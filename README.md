# OCR
Optical Character Recognition for Binary Images Using SIFT Descriptors.
The output of the algorithm is a .json file containing the bounding box of all the characters of a testcase and the character they are matched with. The actual result is also provided which can be compared with the algorithm's result to calculate the F1 score.
The format of the json file is:
                              {"bbox":[xmin,ymin,w,h], "name":"Name of match or UNKNOWN"}
