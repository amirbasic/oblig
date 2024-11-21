import torch
import argparse
from generic_models import FeedForwardNN
from metrics import *
from preprocess_text_generate_features import *
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_dataloader, val_dataloader, epochs, optimizer, criterion):
    """ Train a neural network based on the model provided.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters, FeedForwardNN is used.
        train_dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.
        val_dataloader: torch.train_dataloader: torch.utils.data.DataLoader - same input parametrs as train_dataloader, except the dataset used.
        epochs: int - Number of runs to train the model.
        optimizer: torch.optim - Optimzer method, AdamW is used.
        criterion: torch.nn - Loss function to be used, CrossEntropyLoss is used.

        Returns
        -------
        model: torch.generic_models - Returns the trained model, here it's a trained FeedForwardNN
        history: dictionary - containing performance metrics of the model during training. Key-values are: "accuracy_train", "loss_train", "accuracy_val", "loss_val".

    """
    # Creating a dictionary containing performance metrics
    history = {"accuracy_train": [], "loss_train": [],
               "accuracy_val": [], "loss_val": []}

    # Number of epocs to train the model
    for epoch in range(epochs):
        accuracy_train = 0
        accuracy_val = 0

        # Train model for each step
        for inputs, labels in train_dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            outputs = model(x)
            loss_train = criterion(outputs, y)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            correct = torch.eq(y, predictions).sum().item()
            accuracy_train += (correct / len(predictions)) * 100

        with torch.inference_mode():
            model.eval()
            for inputs, labels in val_dataloader:
                x = inputs.to(device)
                y = labels.to(device)

                outputs = model(x)

                loss_validate = criterion(outputs, y)

                _, predictions = torch.max(outputs, 1)
                correct = torch.eq(y, predictions).sum().item()
                accuracy_val += (correct / len(predictions)) * 100

        print(f'epoch: {epoch+1} / {epochs}, train_loss = {loss_train.item():.4f}, train_accuracy = {accuracy_train/len(train_dataloader):.4f}%, validation_loss = {loss_validate.item():.4f}, validation_accuracy = {accuracy_val/len(val_dataloader):.4f}%')
        history["accuracy_train"].append(accuracy_train/len(train_dataloader))
        history["loss_train"].append(loss_train.item())
        history["accuracy_val"].append(accuracy_val/len(val_dataloader))
        history["loss_val"].append(loss_validate.item())

    return model, history


def evaluate(model, X, y):
    """ Evaluate and return performance metrics of a trained model and given data inputs.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters
        X: torch.tensor - Input labels
        y: torch.tensor - Output labels

        Returns
        -------FeedForwardNN
        model: torch.generic_models - Returns the trained model
        history: dictionary - containing performance metrics of the model during training. Key-values are: "accuracy_train", "loss_train", "accuracy_val", "loss_val".
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
    model_f1_score_macro = f1_score_macro(model, dataloader)
    model_ce_loss = ce_loss(model, dataloader)
    model_recall_score_macro = recall_score_macro(model, dataloader)
    model_precision_score_macro = precision_score_macro(model, dataloader)
    res["model_accuracy"] = model_accuracy
    res["model_f1_score_macro"] = model_f1_score_macro
    res["model_ce_loss"] = model_ce_loss
    res["model_recall_score_macro"] = model_recall_score_macro
    res["model_precision_score_macro"] = model_precision_score_macro

    return res


def time_and_performance(n_hidden_layers=[1, 2, 4, 8, 16]):
    """ Evaluate and return performance metrics using different number of layers in the model. 

        Parameters
        ----------
        n_hidden_layers: List[int] - list of integers containing the number of layers to try for each model.

        Returns
        -------
        returns the plots

    """
    time_train = []
    performance_train = []
    performance_val = []
    f1_train = []
    f1_val = []
    loss_train = []
    loss_val = []

    for n in n_hidden_layers:

        model = FeedForwardNN(input_size=len(vocab),
                              hidden_size=100,
                              hidden_layers=n,
                              num_classes=num_classes
                              ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        model.apply(weight_reset)

        print(f"Training and evaluating {model_name} with {n} hidden layers:")
        start = time.time()

        model.train()
        trained_model, history = train(
            model, train_dataloader, val_dataloader, epochs, optimizer, criterion)

        end = time.time()
        elapsed = end - start

        res = evaluate(trained_model, X_val, y_val)
        res_train = evaluate(trained_model, X_train, y_train)

        acc_train = accuracy(trained_model, train_dataloader)
        acc_val = accuracy(trained_model, val_dataloader)

        f1_t = f1_score_macro(trained_model, train_dataloader)
        f1_v = f1_score_macro(trained_model, val_dataloader)

        time_train.append(elapsed)
        performance_train.append(acc_train)
        performance_val.append(acc_val)
        f1_train.append(f1_t)
        f1_val.append(f1_v)
        loss_train.append(res_train["model_ce_loss"])
        loss_val.append(res["model_ce_loss"])

    hidden_min = min(n_hidden_layers)
    hidden_max = max(n_hidden_layers) + 1

    # time plot
    plt.figure(figsize=(20, 14))
    plt.subplot(1, 4, 1)
    plt.title("Time")
    plt.ylabel("Time")
    plt.xlabel("Hidden layers")
    plt.plot(n_hidden_layers, time_train, "-ok")
    plt.xticks(range(hidden_min, hidden_max))
    # accuracy plot
    plt.subplot(1, 4, 2)
    plt.title("Performance: Accuracy")
    plt.ylabel("Accuracy %")
    plt.xlabel("Hidden layers")
    plt.plot(n_hidden_layers, performance_train,
             "-or", label="training accuracy")
    plt.plot(n_hidden_layers, performance_val,
             "-og", label="validation accuracy")
    plt.xticks(range(hidden_min, hidden_max))
    # f1 plot
    plt.subplot(1, 4, 3)
    plt.title("Performance: f1")
    plt.ylabel("F1 %")
    plt.xlabel("Hidden layers")
    plt.plot(n_hidden_layers, f1_train, "-or", label="training f1")
    plt.plot(n_hidden_layers, f1_val, "-og", label="validation f1")
    plt.xticks(range(hidden_min, hidden_max))
    # loss
    plt.subplot(1, 4, 4)
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("Hidden layers")
    plt.plot(n_hidden_layers, loss_train, "-or", label="Training")
    plt.plot(n_hidden_layers, loss_val, "-og", label="Validation")
    plt.xticks(range(hidden_min, hidden_max))

    plt.legend(loc="upper right")
    plt.savefig(f"{model_name} - Time_and_performance.png")
    plt.show()

    return


def best_model_eval(n_train_and_eval=3):
    """ Prints performance metrics for "n_train_and_eval" runs of the best model.

        Parameters
        ----------
        n_train_and_eval: int - Number of runs to evaluate the best model

        Returns
        -------
        Prints Validation accuracy, recall, precision and F1 along with their respective mean and standard deviation of the "n_train_and_eval" of runs.
    """

    acc_list = []
    recall_list = []
    prec_list = []
    f1_list = []

    for i in range(n_train_and_eval):

        model.apply(weight_reset)

        print("...")
        print(f"Training the model: Model training {i + 1}")

        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=0.001)

        # train function here
        model.train()
        trained_model, history = train(
            model, train_dataloader, val_dataloader, epochs, optimizer, criterion)

        res = evaluate(trained_model, X_val, y_val)

        acc_list.append(round(res["model_accuracy"], 2))
        recall_list.append(round(res["model_recall_score_macro"], 2))
        prec_list.append(round(res["model_precision_score_macro"], 2))
        f1_list.append(round(res["model_f1_score_macro"], 2))

        print(f"Run {i + 1} done")

    acc_mean = np.mean(acc_list)
    acc_sd = np.std(acc_list)

    recall_mean = np.mean(recall_list)
    recall_sd = np.std(recall_list)

    prec_mean = np.mean(prec_list)
    prec_sd = np.std(prec_list)

    f1_mean = np.mean(f1_list)
    f1_sd = np.std(f1_list)

    print(f"Evaluation on our best model over {n_train_and_eval} runs:")
    print("==================================================================")
    print(f"Validation accuracy: {acc_list}")
    print(f"Accuracy mean: {acc_mean:.3f}%, with sd: {acc_sd:.3f}")
    print("------------------------------------------------------------------")
    print(f"Validation recall: {recall_list}")
    print(f"Recall mean: {recall_mean:.3f}, with sd: {recall_sd:.3f}")
    print("------------------------------------------------------------------")
    print(f"Validation Precision: {prec_list}")
    print(f"Precision mean: {prec_mean:.3f}, with sd: {prec_sd:.3f}")
    print("------------------------------------------------------------------")
    print(f"Validation F1: {f1_list}")
    print(f"F1-macro mean: {f1_mean:.3f}, with sd: {f1_sd:.3f}")

    return


def weight_reset(m):
    """ Resets the parameters of a model to avoid starting new training from already pre-trained parameters. 

        Parameters
        ----------
        m: torch.generic_models - with pre-determined input parameters.

    """
    if isinstance(m, nn.Linear):
        m.reset_parameters()


class generator(torch.utils.data.Dataset):
    "Torch.generator - see torch documentation"

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        return len(self.inputs)


if __name__ == "__main__":
    "Creating a interactive CLI to run our program directly"
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "-d", "--data_path",
        help="Directory containing data",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    parser.add_argument(
        "-m", "--max_features",
        help="Upper limit on number of features",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    parser.add_argument(
        "-n", "--model_name",
        help="Name of model",
        type=str,
        required=True,
        metavar="<str>",
        action="store"
    )
    parser.add_argument(
        "-t", "--evaluate_time",
        help="Specify to look at how number of hidden layers affect training time",
        type=bool,
        required=False,
        metavar=bool,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-i", "--ignore_words",
        help="If false add list of ignored words to BoW",
        type=bool,
        required=False,
        metavar=bool,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "-b", "--best_model",
        help="If true train and evaluate the model 3 times and save the trained model",
        type=bool,
        required=False,
        metavar=bool,
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    path_to_data = args.data_path
    max_features = int(args.max_features)
    model_name = args.model_name
    evaluate_time = args.evaluate_time
    ignore_words_bool = args.ignore_words
    best_model_bool = args.best_model

    print("Preprocessing data")
    X_train, X_val, y_train, y_val, vocab, num_classes = preprocess_data(
        path_to_data, ignore_words_bool, max_features, random_state=42)
    print("Preprocessing complete")

    train_dataset = generator(X_train, y_train)
    val_dataset = generator(X_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # Creating our Feed-Forward-NN
    model = FeedForwardNN(input_size=len(
        vocab), hidden_size=100, hidden_layers=1, num_classes=num_classes).to(device)
    # Setting optimizer method to AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Setting our loss function to CrossEntropy
    criterion = nn.CrossEntropyLoss()
    # Number of runs to train our model
    epochs = 4

    print("Training model")
    trained_model, history = train(
        model, train_dataloader, val_dataloader, epochs, optimizer, criterion)

    evaluation_results = evaluate(trained_model, X_val, y_val)

    print(evaluation_results)

    if best_model_bool:
        print("best model")
        best_model_eval()
        torch.save(model.state_dict(), model_name+".pth")

    time_and_performance()
