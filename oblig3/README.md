IN5550 - Spring 2023 Mandatory assignment 3 Named entity recognition and pre-trained language models

Cornelius Bencsik, Amir Basic & Torstein Forseth corneb, amirbas & torfor

Git repository: https://github.uio.no/torfor/IN5550/tree/main/oblig3 Fox directory: ../../../projects01/ec30/CocoAmirTorfor

Instructions:

To run our best model, our intention is that you are in the CocoAmirTorfor
folder referenced above. There you will run a slurm file that run the 
"predict_on_test.py" file, which use "best_model_ass2.bt" as the best model.
The path to that model is, from CocoAmirTorfor, "models/best_model_ass3.pt".

Since you run the slurm file from CocoAmirTorfor, to access the predict_on_test.py
file, you should therefore run the following ssh in the slurm file you use:

    python3 IN5550/oblig3/predict_on_test.py --test <testfile>
output: The output will be:

    1. A slurm-xxxxxx.out file containing the evaluation run from predict_on_test.py
    with the accuracy obtained

    2. A predictions.conlly.gz file containing the original premise and hypothesis from the 
    <testfile> where the labels are replaced with the predicted labels from our model

Arguments:

--test: <testfile>, default = "IN5550/oblig3/data/gold_label_val.conllu.gz"
default value is the path to the original validation dataset that we created, but here you should put the path
to your own testfile

--model_path: <modelfile>, default = "models/best_model_ass3.pt"
default value is the path to our best model from CocoAmirTorfor on fox. Therefore we recommend 
running from CocoAmirTorfor, if not, this path has to be adapted

--output_path", default = "predictions.conllu.gz"
name of the desired output file