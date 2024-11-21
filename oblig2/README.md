IN5550 - Spring 2023
Mandatory assignment 2
Word Embeddings and Recurrent Neural Networks

Cornelius Bencsik, Amir Basic & Torstein Forseth
corneb, amirbas & torfor

Git repository: https://github.uio.no/torfor/IN5550/tree/main/oblig2
Fox directory: ../../../projects01/ec30/CocoAmirTorfor

Instructions:

    To run our best model, our intention is that you are in the CocoAmirTorfor
    folder referenced above. There you will run a slurm file that run the 
    "predict_on_test.py" file, which use "best_model_ass2.bt" as the best model.
    The path to that model is, from CocoAmirTorfor, "models/best_model_ass2.bt".

    Since you run the slurm file from CocoAmirTorfor, to access the predict_on_test.py
    file, you should therefore run the following ssh in the slurm file you use:

        python3 IN5550/oblig2/predict_on_test.py --test <testfile>


output:
    The output will be:

        1. A slurm-xxxxxx.out file containing the evaluation run from predict_on_test.py
        with the accuracy obtained

        2. A predictions.tsv file containing the original premise and hypothesis from the 
        <testfile> where the labels are replaced with the predicted labels from our model

        3. A predictions.tsv.gz file containing the original premise and hypothesis from the 
        <testfile> where the labels are replaced with the predicted labels from our model

Arguments:

    --test: <testfile>, default = "IN5550/oblig2/data/mnli_train.tsv.gz"
    default value is the path to the original training dataset, but here you should put the path
    to your own testfile

    --model_path: <modelfile>, default = "models/best_model_ass2.pt"
    default value is the path to our best model from CocoAmirTorfor on fox. Therefore we recommend 
    running from CocoAmirTorfor, if not, this path has to be adapted