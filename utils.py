import torch

def compute_fluctuations(models, test_loader):
    predictions = []
    with torch.no_grad():
        for model in models:
            model.eval()
            preds = []
            for data, _ in test_loader:
                data = data.view(data.size(0),-1)
                output = model(data)
                preds.append(output)
            predictions.append(torch.cat(preds))
    predictions = torch.stack(predictions)
    mean_prediction = predictions.mean(dim=0)
    fluctuations = torch.sqrt(((predictions - mean_prediction) ** 2).mean(dim=0))
    return fluctuations.mean().item()


def ensemble_predictions(models, test_loader):
    with torch.no_grad():
        ensemble_preds = None
        for model in models:
            model.eval()
            preds = []
            for data, _ in test_loader:
                data = data.view(data.size(0),-1)
                output = model(data)
                preds.append(output)
            preds = torch.cat(preds)

            if ensemble_preds is None:
                ensemble_preds = preds
            else:
                ensemble_preds += preds
    return ensemble_preds/len(models)