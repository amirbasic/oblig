import argparse
from generic_models import FeedForwardNN
from preprocess_text_generate_features import * 
from main import generator, evaluate
from metrics import *

def evaluate_test(model, X, y):
    """ Evaluate a model performance based on given input and output data.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters, FeedForwardNN is used.
        X: array - input data for predicting labels.
        y: array - actual output labels.

        Returns
        -------
        res: dictionary - containing performance metrics of the model based on new data. 
                          Key-values are: "model_accuracy", "model_f1_score_each_class", "model_ce_loss", "model_recall_score_each_class", "model_precision_score_each_class"
    """
    res = {}
    model.eval()
    dataset = generator(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model_accuracy = accuracy(model, dataloader)
    model_f1_score_each_class = f1_score_each_class(model, X, y)
    model_ce_loss = ce_loss(model, dataloader)
    model_recall_score_each_class = recall_score_each_class(model, X, y)
    model_precision_score_each_class = precision_score_each_class(model, X, y)
    res["model_accuracy"] = model_accuracy; res["model_f1_score_each_class"] = model_f1_score_each_class; res["model_ce_loss"] = model_ce_loss; res["model_recall_score_each_class"] = model_recall_score_each_class; res["model_precision_score_each_class"] = model_precision_score_each_class;
    
    return res 
    
if __name__ == "__main__":
    "Creating a interactive CLI to run our program directly"
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "-d", "--data_path",
        help="Directory containing test data",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    parser.add_argument(
        "-m", "--model_path",
        help="Path to pretrained model",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    parser.add_argument(
        "-t", "--train_data",
        help="Path to original training data",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    args = parser.parse_args()
    
    path_to_test_data = args.data_path
    pretrained_model_path = args.model_path
    path_to_train_data = args.train_data
    
    X_test, y_test, vocab, num_classes = preprocess_data(path_to_train_data, path_to_test_data = path_to_test_data, random_state=42, evaluate=True)
    
    model = FeedForwardNN(input_size=len(vocab), hidden_size=100, hidden_layers=1, num_classes=num_classes)
    model.load_state_dict(torch.load(pretrained_model_path))
    
    res = evaluate_test(model, X_test, y_test)
    res2 = evaluate(model, X_test, y_test)
    
    print(res)
    print(res2)