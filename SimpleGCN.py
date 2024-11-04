import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve,auc
)
from collections import defaultdict
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Setting the seed for an consistent result
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# selecting the device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## Model

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul( input, self.weight)
        output = torch.einsum('bij,bjd->bid', [adj, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNModel(nn.Module):
    def __init__(self, in_features, out_features, num_classes=2):
        super(GCNModel, self).__init__()
        self.gc = GraphConvolution(in_features, out_features)  # Single GCN layer
        #self.batch_norm = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(out_features, num_classes)

    def forward(self, input, adj):
        x= self.gc(input, adj)
        #x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gc(input, adj).mean(dim=1)  # Mean pooling over nodes
        x = self.fc(x)
        return x

# ## uncoment to get the model summary
# from torchsummary import summary
# model = SimpleGCNModel(in_features=150, out_features=64, num_classes=2).to(device)
# summary(model, input_size=[(1, 150, 150), (1, 150, 150)])



## DATA creation template
class CreateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.ad_folders = [os.path.join(root_dir, f'AD{i}') for i in range(1, 11)]
        self.cn_folders = [os.path.join(root_dir, f'CN{i}') for i in range(1, 11)]
        self.all_folders = self.ad_folders + self.cn_folders
        self.labels = [1] * len(self.ad_folders) + [0] * len(self.cn_folders)  # 1 for AD, 0 for CN
        self.transform = transform

    def __len__(self):
        return len(self.all_folders)

    def __getitem__(self, idx):
        folder_path = self.all_folders[idx]
        label = self.labels[idx]
        
        # functional and structural connectivity matrices from text files
        fc_path = os.path.join(folder_path, 'FunctionalConnectivity.txt')
        sc_path = os.path.join(folder_path, 'StructuralConnectivity.txt')
        
        # creating matrices from text files
        fc_matrix = np.loadtxt(fc_path)
        sc_matrix = np.loadtxt(sc_path)
        fc_matrix = (fc_matrix - np.mean(fc_matrix)) / np.std(fc_matrix)  # Z-score normalization
        sc_matrix = (sc_matrix - np.mean(sc_matrix)) / np.std(sc_matrix)
        
        sample = {
            'fc_matrix': torch.tensor(fc_matrix, dtype=torch.float32), 
            'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 
            'label': torch.tensor(label, dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

## Evaluation metrics
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "Accuracy": acc, "Precision": precision, "Recall": recall, "F1-Score": f1,
        "Balanced Accuracy": balanced_acc, "MCC": mcc, "AUC-ROC": auc, "Confusion Matrix": cm
    }


#Data split and Loading
dataset = CreateDataset(root_dir='6389Project2Data')
labels = [dataset[i]['label'].item() for i in range(len(dataset))]

train_val_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=0.2,  # 20% as a separate test set
    stratify=labels,
    random_state=42
)

test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=6)

# k-Fold Cross-Validation on train+validation data
k_folds = 4
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# separate labels for train+validation split
train_val_labels = [labels[i] for i in train_val_indices]


## Training-----------------------------------------------------------------
# Storing results for each fold
fold_results = defaultdict(list)
epoch_train_loss = []
epoch_val_loss = []

# Iterating over each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
    print(f"\nFold {fold+1}/{k_folds}")
    
    # Creating train and validation subsets using the indices from skf.split
    train_subset = Subset(dataset, [train_val_indices[i] for i in train_idx])
    val_subset = Subset(dataset, [train_val_indices[i] for i in val_idx])
    
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=4)
    
    # Initialize model, optimizer, and loss function
    model = GCNModel(in_features=150, out_features=25, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-2)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    fold_train_loss = []
    fold_val_loss = []
    best_val_loss = float('inf')
    for epoch in range(10):
        model.train()
        train_loss = 0
        all_train_labels = []
        all_train_preds = []
        for batch in train_loader:
            fc_matrix = batch['fc_matrix'].to(device)
            sc_matrix = batch['sc_matrix'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(fc_matrix, sc_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            _, train_preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(train_preds.cpu().numpy())
            
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                fc_matrix = batch['fc_matrix'].to(device)
                sc_matrix = batch['sc_matrix'].to(device)
                labels = batch['label'].to(device)

                outputs = model(fc_matrix, fc_matrix)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Storing losses for this epoch
        fold_train_loss.append(avg_train_loss)
        fold_val_loss.append(avg_val_loss)
        
        # Calculating metrics
        fold_metrics = calculate_metrics(all_labels, all_preds)
        fold_metrics["Validation Loss"] = avg_val_loss  # Store validation loss in the metrics
        
        print(f"Epoch [{epoch+1}/10], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # computing best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Saving best model for this fold
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
    
    # Storing results for this fold
    for key, value in fold_metrics.items():
        fold_results[key].append(value)
        
    # Appending fold results
    epoch_train_loss.append(fold_train_loss)
    epoch_val_loss.append(fold_val_loss)

# printing average metrics across folds
print("\nAverage Cross-Validation Results:")
for metric, values in fold_results.items():
    avg_metric = np.mean([v for v in values if v is not None])  # Handling None in AUC-ROC
    print(f"{metric}: {avg_metric:.4f}")


# Converting lists to numpy arrays for easy averaging
epoch_train_loss = np.array(epoch_train_loss)
epoch_val_loss = np.array(epoch_val_loss)

# Calculating mean across folds for each epoch
mean_train_loss = np.mean(epoch_train_loss, axis=0)
mean_val_loss = np.mean(epoch_val_loss, axis=0)

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plotting Training Loss
color = 'tab:blue'
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color=color)
ax1.plot(mean_train_loss, label="Training Loss", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel("Validation Loss", color=color)
ax2.plot(mean_val_loss, label="Validation Loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle("Average Training and Validation Loss Across Folds with Dual Y-Axes")
fig.tight_layout()
plt.savefig("./results/Average_Training_and_Validation_Loss.png")

##-----------------------------------------------------------------------------------------------------

## Testing and evaluation
best_fold = np.argmin(fold_results["Validation Loss"]) + 1  # +1 because fold numbers are 1-indexed
best_model = GCNModel(in_features=150, out_features=25, num_classes=2).to(device)
best_model.load_state_dict(torch.load(f'best_model_fold_{best_fold}.pth'))

test_labels = []
test_preds = []
best_model.eval()
with torch.no_grad():
    for batch in test_loader:
        fc_matrix = batch['fc_matrix'].to(device)
        sc_matrix = batch['sc_matrix'].to(device)
        labels = batch['label'].to(device)

        outputs = best_model(fc_matrix, sc_matrix)
        _, preds = torch.max(outputs, 1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

# Calculating test metrics
test_metrics = calculate_metrics(test_labels, test_preds)
print("\nTest Set Results:")
print(test_metrics)

# Getting probability scores
test_probs = []
with torch.no_grad():
    for batch in test_loader:
        fc_matrix = batch['fc_matrix'].to(device)
        sc_matrix = batch['sc_matrix'].to(device)
        
        outputs = best_model(fc_matrix, sc_matrix)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for the positive class AD
        test_probs.extend(probs.cpu().numpy())

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve")
plt.legend(loc="lower right")
#plt.show()
plt.savefig("./results/AUC_ROC.png")

# confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Plotting
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("./results/confusion_matrix.png")


