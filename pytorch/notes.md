=========================================
################ PYtorch ################
=========================================

```python
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Inside train loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
# Save the model checkpoint    
torch.save(model.state_dict(), 'model.ckpt')
```

