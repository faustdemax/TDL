import torch

def compute_fluctuations(predictions):
    print("[INFO] Computing fluctuations...")
    predictions = torch.stack(predictions)  
    mean_prediction = predictions.mean(dim=0) 
    fluctuations = torch.sqrt(((predictions - mean_prediction) ** 2).mean(dim=0)) 
    return fluctuations.mean().item()

def ensemble_predictions(models, test_loader):
    print("[INFO] Computing ensemble predictions...")
    with torch.no_grad():
        ensemble_preds = None
        for model in models:
            model.eval()
            all_preds = []
            for batch_x, _ in test_loader:
                preds = model(batch_x)
                all_preds.append(preds)
            preds = torch.cat(all_preds)

            if ensemble_preds is None:
                ensemble_preds = preds
            else:
                ensemble_preds += preds

        return ensemble_preds / len(models) 