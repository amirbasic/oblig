import sklearn.metrics
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('always')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f1_score_macro(model, dataloader):
    """ Calculates the average F1 score of a model and a given dataset

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.

        Returns
        -------
        avg_f1_score: float - average F1 score

    """
    with torch.inference_mode():
        model.eval()
        f1_score = 0
        for inputs, labels in dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            f1_score += sklearn.metrics.f1_score(y,
                                                 predictions, average='macro')
            avg_f1_score = f1_score/len(dataloader)

        return avg_f1_score


def accuracy(model, dataloader):
    """ Calculates the average accuracy score of a model and a given dataset

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.

        Returns
        -------
        average_accuracy_score: float - average accuracy score
    """
    with torch.inference_mode():
        model.eval()
        accuracy = 0
        for inputs, labels in dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            correct = torch.eq(y, predictions).sum().item()
            accuracy += (correct / len(predictions)) * 100
            average_accuracy_score = accuracy/len(dataloader)

        return average_accuracy_score


def ce_loss(model, dataloader):
    """ Calculates the average cross entropy loss of a model and a given dataset

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.

        Returns
        -------
        average_ce_loss: float - average cross entropy loss
    """
    with torch.inference_mode():
        loss_fn = nn.CrossEntropyLoss()
        model.eval()
        loss = 0
        for inputs, labels in dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            y = y.type(torch.LongTensor)

            outputs = model(x)
            loss += loss_fn(outputs, y)
            average_ce_loss = loss.item()/len(dataloader)

        return average_ce_loss


def recall_score_macro(model, dataloader):
    """ Calculates the average recall score of a model and a given dataset

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.

        Returns
        -------
        average_recall_score: float - average recall score
    """
    with torch.inference_mode():
        model.eval()
        recall_score = 0
        for inputs, labels in dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            recall_score += sklearn.metrics.recall_score(
                y, predictions, average='macro')
            average_recall_score = recall_score/len(dataloader)
        return average_recall_score


def precision_score_macro(model, dataloader):
    """ Calculates the average precision score of a model and a given dataset

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        dataloader: torch.utils.data.DataLoader - with pre-determined input parametrs.

        Returns
        -------
        average_precision_score: float - average precision score
    """
    with torch.inference_mode():
        model.eval()
        precision_score = 0
        for inputs, labels in dataloader:
            x = inputs.to(device)
            y = labels.to(device)

            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            precision_score += sklearn.metrics.precision_score(
                y, predictions, average='macro')
            average_precision_score = precision_score/len(dataloader)

        return average_precision_score


def precision_score_each_class(model, X, y):
    """ Calculates the precision score of each class in a model.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        X: array - input data for predicting labels.
        y: array - actual output labels.

        Returns
        -------
        precision_score: float - precision score
    """
    with torch.inference_mode():
        model.eval()

        X = torch.from_numpy(X)

        outputs = model(X)
        _, predictions = torch.max(outputs, 1)
        precision_score = sklearn.metrics.precision_score(
            y, predictions, average=None)

        return precision_score


def recall_score_each_class(model, X, y):
    """ Calculates the recall score of each class in a model.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        X: array - input data for predicting labels.
        y: array - actual output labels.

        Returns
        -------
        recall_score: float - recall score
    """
    with torch.inference_mode():
        model.eval()

        X = torch.from_numpy(X)

        outputs = model(X)
        _, predictions = torch.max(outputs, 1)
        recall_score = sklearn.metrics.recall_score(
            y, predictions, average=None)

        return recall_score


def f1_score_each_class(model, X, y):
    """ Calculates the F1 score of each class in a model.

        Parameters
        ----------
        model: torch.generic_models - with pre-determined input parameters.
        X: array - input data for predicting labels.
        y: array - actual output labels.

        Returns
        -------
        f1_score: float - f1 score
    """
    with torch.inference_mode():
        model.eval()

        X = torch.from_numpy(X)

        outputs = model(X)
        _, predictions = torch.max(outputs, 1)
        f1_score = sklearn.metrics.f1_score(y, predictions, average=None)

        return f1_score
