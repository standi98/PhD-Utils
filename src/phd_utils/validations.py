import pandas as pd
import torch

def validate_model_output(model, dataset, batch_size=32):
    sample_X, sample_y = dataset[0]
    
    # Handle both scalar and vector targets
    dummy_batch_X = sample_X.unsqueeze(0).expand(batch_size, -1)
    
    if sample_y.dim() == 0:  # scalar target
        dummy_batch_y = sample_y.unsqueeze(0).expand(batch_size)
    else:                     # vector target
        dummy_batch_y = sample_y.unsqueeze(0).expand(batch_size, -1)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_batch_X)
    
    if output.shape != dummy_batch_y.shape:
        raise ValueError(
            f"Shape mismatch: model output {output.shape} != target {dummy_batch_y.shape}.\n"
            f"Consider adding squeeze/unsqueeze to the model's forward method."
        )
