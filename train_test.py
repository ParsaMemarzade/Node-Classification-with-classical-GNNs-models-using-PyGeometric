import torch

def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()  
    out = model(data.x, data.edge_index)  
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  
    loss.backward()  
    optimizer.step()  
    return loss

def test(model, mask, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      correct = pred[mask] == data.y[mask]  
      acc = int(correct.sum()) / int(mask.sum()) 
      return acc
