# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from TernaryClassifier import TernaryClassifier

def load_data(filename):
    return pd.read_csv(filename)

def create_dfs(df, train_samples=20):
    """ Concatenate training and testing data into seperate data frames """
    # "Class" column in the .csv file refers to its label
    # Split into 3 data frames based on classifier
    fast = df[df["class"] == 0]
    normal = df[df["class"] == 1]
    careful = df[df["class"] == 2]
    
    # train_samples refers to how many samples in a specific classifier will be used for training
    # (the remaining will be used for testing)
    
    # Concatinate the training data into one data frame, shuffle data set randomly
    train_df = pd.concat([fast[:train_samples],
                         normal[:train_samples], 
                         careful[:train_samples]]).sample(frac=1).reset_index(drop=True)
    test_df = pd.concat([fast[train_samples:], 
                        normal[train_samples:], 
                        careful[train_samples:]]).sample(frac=1).reset_index(drop=True)
    return train_df, test_df

def preprocess_data(train_df, test_df, class_col="class"):
    """ Convert data frames to numpy arrays, isolate classifier column"""
    # Create new dataframe excluding the "class" column, and convert to numpy tensor
    dvar_train = train_df.drop(columns=[class_col]).values.astype("float32")
    dvar_test = test_df.drop(columns=[class_col]).values.astype("float32")
    
    # Create new dataframe with just the "class" column, convert to numpy tensor
    classifier_train = train_df[class_col].values.astype("int64")
    classifier_test = test_df[class_col].values.astype("int64")
    
    # Scale dependent variables so mean = 0, stdev = 1
    scaler = StandardScaler()
    dvar_train = scaler.fit_transform(dvar_train)
    dvar_test = scaler.transform(dvar_test)
    
    return dvar_train, classifier_train, dvar_test, classifier_test

def create_data_loaders(x_train, y_train, x_test, y_test, batch_size=16):
    """Create PyTorch DataLoader objects for training and testing."""
    # Concatonate into a tensor
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

def initialize_model(input_dim, device):
    """ Create model using # of dependent vars, specify use of the cpu"""
    model = TernaryClassifier(input_dim)
    return model.to(device)

def train_classifier(model, train_loader, device, learning_rate=0.001, epochs=100):
    """ Train the model """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # batch_x = input dependent vars
        # batch_y = classifier labels
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) # Transfer data to cpu
            
            # Clear the previous gradients, as they accumulate
            optimizer.zero_grad()
            # Forward pass, pass features through the layers of the nn
            outputs = model(batch_x)
            # Compare model's predictions (outputs) to the true labels (batch_y) using crossentropy
            loss = criterion(outputs, batch_y)
            # Backpropagation
            loss.backward()
            # Update model's parameters using the computed gradients
            optimizer.step()
            # Get the total loss for each epoch
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {round(total_loss,4)}")

def evaluate_model(model, test_loader, device):
    """ Test the models performance """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(): # Don't track gradients
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x) # Output predictions for batch e.g [0.2,0.5,0.3]
            _, predicted = torch.max(outputs, 1) # choose class with highest score
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {round(accuracy,2)}%")
    return accuracy

def main():
    # Config
    filename = "corpus.csv"
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    train_samples_per_class = 20
    device = torch.device("cpu")
    
    # Data pipeline
    df = load_data(filename)
    train_df, test_df = create_dfs(df, train_samples_per_class)
    dvar_train, classifier_train, dvar_test, classifier_test = preprocess_data(train_df, test_df)
    train_loader, test_loader = create_data_loaders(dvar_train, classifier_train, dvar_test, classifier_test, batch_size)
    
    # Model Initialization
    input_dim = dvar_train.shape[1] # 2nd dimension is equivalent to # of dependent variables
    model = initialize_model(input_dim, device) # Create instance of the TernaryClassifier class
    
    # Train
    train_classifier(model, train_loader, device, learning_rate, epochs)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
