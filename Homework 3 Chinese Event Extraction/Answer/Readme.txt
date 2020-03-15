CRF++ - 0.58£º
      I store the training set in "CRF++-0.58\example"
      You can open the terminal in 'CRF++-0.58' and enter the instructions as
      "crf_learn -f 3 -c 1.5 ./example/Argument/template   ./example/Argument/argument_train.data   ./example/Argument/model.data" 
      "crf_test -m ./example/Argument/model.data  ./example/Argument/argument_test.data > ./example/Argument/argument_result.txt"      

       If you want to train the Trigger dataset, you can enter
       "crf_learn -f 3 -c 1.5 ./example/Trigger/template   ./example/Trigger/trigger_train.data   ./example/Trigger/model.data" 
       "crf_test -m ./example/Trigger/model.data  ./example/Trigger/trigger_test.data > ./example/Trigger/trigger_result.txt"

       Then the text "trigger_result.txt" and "argument_result.txt" will be generated, you can use the "eval.py" in each files to evaluate it.

Homework3:
       DataLoader.py : The python program to load data for HMM training
       extraction.py   :  The HMM model, you can adjust the parameters to select the training set and hyperparameters and directly run
                                 It can generate the "_result.txt" and perform automatic evaluation
       trigram_extraction.py   :  The trigram-HMM model, you can adjust the parameters to select the training set and hyperparameters and directly run
                                 It can generate the "Trigram_result.txt" and perform automatic evaluation

Chinese Event Extraction.pdf:  The report for the project

Feel free to contact Cheng Zhong (16307110259@fudan.edu.cn) for further information
             
       
