import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix




from sklearn.preprocessing import StandardScaler




# =============================================================================
# MTL logistc regression model
# =============================================================================
class MultiTaskLogisticRegression(nn.Module):
    def __init__(self, input_size, num_tasks, num_classes_per_task):
        super(MultiTaskLogisticRegression, self).__init__()
        #self.shared = nn.Linear(input_size, 128)
        self.tasks = nn.ModuleList([
            nn.Linear(input_size, num_classes_per_task[i]) for i in range(num_tasks)
        ])

    def forward(self, x):
        #x = torch.relu(self.shared(x))
        return [task(x) for task in self.tasks]


# =============================================================================
# MTL MLP model shared layer
# =============================================================================

class MultiTaskSharedModel(nn.Module):
    def __init__(self, input_size, num_tasks, num_classes_per_task):
        super(MultiTaskSharedModel, self).__init__()
        # Shared layers
        self.shared_layer1 = nn.Linear(input_size, 128)
        #self.shared_layer2 = nn.Linear(128, 64)

        # Task-specific layers
        self.tasks = nn.ModuleList([
            nn.Linear(128, num_classes_per_task[i]) for i in range(num_tasks)
        ])

    def forward(self, x):
        # Applying shared layers
        x = torch.relu(self.shared_layer1(x))
        #x = torch.relu(self.shared_layer2(x))

        # Applying task-specific layers
        outputs = [task(x) for task in self.tasks]
        return outputs

def elastic_net_penalty(model, alpha, l1_ratio):
    l1_penalty, l2_penalty = 0., 0.
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
        l2_penalty += torch.sum(param ** 2)
    return l1_ratio * alpha * l1_penalty + (1 - l1_ratio) * alpha * l2_penalty
    

def task_coupling_penalty_with_R(model, R, lambda_tc):
    penalty = 0.0
    num_tasks = len(model.tasks)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            # Get the L2 norms of the weights of the tasks
            norm_i = torch.norm(list(model.tasks[i].parameters())[0])
            norm_j = torch.norm(list(model.tasks[j].parameters())[0])
            # Calculate the penalty scaled by the relationship matrix R
            #print(R.shape)

            penalty += R[i, j] * (norm_i - norm_j).pow(2)

    return lambda_tc * penalty

# =============================================================================
# Overall General Optimization criteria
# \begin{align*}
# L &= L_{CE} + L_{EN} + L_{TC} \\
# \text{where} \\
# L_{CE} &= \sum_{i=1}^{T} \text{CrossEntropy}(y_{i}, \hat{y}_{i}) \\
# L_{EN} &= \alpha \left( \rho \cdot ||w||_1 + (1 - \rho) \cdot ||w||_2^2 \right) \\
# L_{TC} &= \lambda \sum_{i=1}^{T}\sum_{j=i+1}^{T} R_{ij} \cdot (||w_i||_2 - ||w_j||_2)^2
# \end{align*}
# =============================================================================

# We can put more emphasis on task 1 (match task)
# L_{CE} = \sum_{i=1}^{7} w_i \cdot \text{CrossEntropy}(y_i, \hat{y}_i)


def custom_loss_fn(outputs, targets, model, alpha, l1_ratio, lambda_tc, R, task_weights):
    loss_fn = nn.CrossEntropyLoss()  # Assuming classification tasks
    ce_losses = [task_weights[i] * loss_fn(outputs[i], targets[i]) for i in range(len(outputs))]
    ce_loss = sum(ce_losses)
    reg_loss = elastic_net_penalty(model, alpha, l1_ratio) 
    coupling_loss = task_coupling_penalty_with_R(model, R, lambda_tc)
    # print(ce_loss)
    # print(reg_loss)
    # print(coupling_loss)
    return ce_loss + reg_loss + coupling_loss


def perform_Kfold_cv_MTL(X_train, X_test, y_train, y_test, seed_value, compute_baseline,
                     enable_plots, hyps_dict):
    rnd_state = seed_value
    
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base0 = base1 = base2 = base3 = base4 = base5 = 0
    #print(X_train)
    X_train=X_train.fillna(999)
    X_test=X_test.fillna(999)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    # apply same transformation to test data
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled=np.nan_to_num(X_train_scaled, nan=999)
    X_test_scaled=np.nan_to_num(X_test_scaled, nan=999)
    
    # X_train_scaled=X_train.to_numpy()
    # X_test_scaled=X_test.to_numpy()
    # =============================================================================
    # Train and Tesing 
    # =============================================================================


    input_size = X_train.shape[1]  # Example input size
    num_tasks = 6  # Number of tasks
    #example task 1 = match
    # relationships / graphs among task can be given by domain expertise 

    #Put more emphasis on the match task
    task_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Example weights

    num_classes_per_task = [2, 2, 2, 2, 2, 2]  # Number of classes for each task
    # Example: Defining a nxn static relationship matrix for n tasks
    # Adjust this matrix based on your specific task relationships
    #relationship --> mean
    # R = torch.tensor([[0.83, -0.17, -0.17, -0.17, -0.17, -0.17],  # Relationship of Task 1 with other tasks
    #                   [-0.17, 0.83,-0.17, -0.17, -0.17, -0.17], #R = eye(num_tasks) - ones(num_tasks)/num_tasks.
    #                   [-0.17, -0.17, 0.83, -0.17, -0.17, -0.17],
    #                   [-0.17, -0.17, -0.17, 0.83, -0.17, -0.17],
    #                   [-0.17, -0.17, -0.17, -0.17, 0.83, -0.17],
    #                   [-0.17, -0.17, -0.17, -0.17, -0.17, 0.83]]
    #                 )

    #relationship based on a priori knowledge e.g. task relations 1
    R = np.array([
        [288, 23, 12, 6, 4, 0],
        [12, 306, 15, 28, 4, 0],
        [6, 5, 722, 12, 1, 1],
        [7, 9, 15, 540, 4, 1],
        [33, 20, 21, 20, 103, 1],
        [0, 0, 0, 4, 0, 17]
    ])
    # R = np.array([
    #     [288, 0, 0, 0, 0, 0],
    #     [0, 306, 0, 0, 0, 0],
    #     [0, 0, 722, 0, 0, 0],
    #     [0, 0, 0, 540, 0, 0],
    #     [0, 0, 0, 0, 103, 0],
    #     [0, 0, 0, 0, 0, 17]
    # ])
    print('Original R matrix')
    print(R)
    # Temporarily set diagonal elements to zero
    np.fill_diagonal(R, 0)
    
    # Calculate the sum of absolute values of the non-diagonal elements for each row
    row_sums = np.sum(np.abs(R), axis=1)
    
    # To avoid division by zero, set zero sums to one (this will affect rows with all zeros)
    row_sums[row_sums == 0] = 1
    
    # Normalize non-diagonal elements and make them negative
    R_normalized = -np.abs(R) / row_sums[:, np.newaxis]
    
    # Set diagonal elements back to one
    np.fill_diagonal(R_normalized, 1)

    # epsilon = 1e-8  # D
    # R_normalized = np.where(R_normalized == 0, epsilon, R_normalized)

    print('Normalized R matrix')
    print(R_normalized)

    R = torch.tensor([R_normalized])
    R = R.squeeze()

    #MLP or LR
    # model = MultiTaskSharedModel(input_size, num_tasks, num_classes_per_task)
    if hyps_dict['model'] == 'MTL-LR':
        model = MultiTaskLogisticRegression(input_size, num_tasks, num_classes_per_task)
    else:
        raise ValueError('Invalid model name.')

    optimizer = optim.Adam(model.parameters(), lr=hyps_dict['lr'])

    # Dummy training data
    #X_train = torch.randn(640, input_size)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    #Y_train = [torch.randint(0, n_classes, (640,)) for n_classes in num_classes_per_task]
    Y_train_tensor = [torch.tensor(y_train[label].values) for label in y_train.columns]

    
    # Dummy testing data
    #X_test = torch.randn(320, input_size)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    #Y_test = [torch.randint(0, n_classes, (320,)) for n_classes in num_classes_per_task]
    Y_test_tensor = [torch.tensor(y_test[label].values) for label in y_test.columns]

    
    #print(outputs) 

    model.train()
    num_epochs=20
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)

        # Calculate and sum losses for each task
        total_loss = custom_loss_fn(outputs, Y_train_tensor, model, alpha=1, l1_ratio=hyps_dict['l1ratio'], lambda_tc=hyps_dict['lambdatc'], R=R, task_weights=task_weights)
        # print('LOSS')
        # print(total_loss)

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        
    model.eval()
    test_accuracies = []
    label_cm=[None] * num_tasks
    y_pred_all=[]
    with torch.no_grad():
        for i in range(num_tasks):
            y_pred = []

            for x in X_test_tensor:
                output = model(x.unsqueeze(0))[i]
                predicted_class = torch.argmax(output)
                y_pred.append(predicted_class.item())

            accuracy = accuracy_score(Y_test_tensor[i].numpy(), y_pred)
            test_accuracies.append(accuracy)
            y_pred_all.append(y_pred)
            label_cm[i]=(confusion_matrix(Y_test_tensor[i].numpy(), y_pred))
    print(f"Test Accuracies for each task: {test_accuracies}")   
    y_pred_all=np.transpose(y_pred_all)    
    #print(label_cm)     
        

    # print(importance.shape())
    #label_cm = multilabel_confusion_matrix(y_test, y_pred)
    
# =============================================================================
#     
#     #CASSETTE
# =============================================================================
    #ypred2=X_test['CASSETTE']
    label_cm2 = None
    if compute_baseline:
        ypred2 = X_test.iloc[:, 0]
        label_cm2 = confusion_matrix(y_test['CASSETTE'], ypred2)
        print('Baseline Cassette')
        print(label_cm2)
        print()
    print('Cassette')
    print(label_cm[0])
    print()
    # print baseline CASSETTE
    # ypred2 = X_test['CASSETTE']
    # label_cm2 = confusion_matrix(y_test['CASSETTE'], ypred2)
    # print('Baseline Cassette')
    # print(label_cm2)
    # print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[0], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target Cassette')
        ax.xaxis.set_ticklabels(['Cassette NO', 'Cassette YES'])
        ax.yaxis.set_ticklabels(['Cassette NO', 'Cassette YES'])
        # plt.savefig('figure2806/cmMatrixCassetteW6.pdf', bbox_inches='tight')
        plt.show()

    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base0 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA Cassette:')
        print("%.3f" % base0)
        print()

    #prediction
    tn, fp, fn, tp = label_cm[0].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba0 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print('BA Cassette:')
    print("%.3f" % ba0)
    print()

    # print baseline CT
    # ypred2 = X_test['CT']
    # label_cm2 = confusion_matrix(y_test['CT'], ypred2)
    # print('Baseline CT')
    # print(label_cm2)
    # print()

# =============================================================================
# #CT
# =============================================================================
    #ypred2=X_test['CT']
    if compute_baseline:
        ypred2 = X_test.iloc[:, 1]
        label_cm2 = confusion_matrix(y_test['CT'], ypred2)
        print('Baseline CT')
        print(label_cm2)
        print()
    print('CT')
    print(label_cm[1])
    print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[1, :, :], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target CT')
        ax.xaxis.set_ticklabels(['CT NO', 'CT YES'])
        ax.yaxis.set_ticklabels(['CT NO', 'CT YES'])
        # plt.savefig('figure2806/cmMatrixCTW6.pdf', bbox_inches='tight')
        plt.show()

    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base1 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA CT:')
        print("%.3f" % base1)
        print()
    #prediction
    tn, fp, fn, tp = label_cm[1].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba1 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba1)
    print()

    # print baseline NE
    # ypred2 = X_test['NE']
    # label_cm2 = confusion_matrix(y_test['NE'], ypred2)
    # print('Baseline NE')
    # print(label_cm2)
    # print()
    
    
# =============================================================================
# #NE
# =============================================================================

    if compute_baseline:
        ypred2 = X_test.iloc[:, 2]
        label_cm2 = confusion_matrix(y_test['NE'], ypred2)
        print('Baseline NE')
        print(label_cm2)
        print()
    print('NE')
    print(label_cm[2])
    print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[2, :, :], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target NE')
        ax.xaxis.set_ticklabels(['NE NO', 'NE YES'])
        ax.yaxis.set_ticklabels(['NE NO', 'NE YES'])
        # plt.savefig('figure2806/cmMatrixNEW6.pdf', bbox_inches='tight')
        plt.show()
    
    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base2 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA NE:')
        print("%.3f" % base2)
        print()
    #predicted
    tn, fp, fn, tp = label_cm[2].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba2 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba2)
    print()

    # print baseline NF
    # ypred2 = X_test['NF']
    # label_cm2 = confusion_matrix(y_test['NF'], ypred2)
    # print('Baseline NF')
    # print(label_cm2)
    # print()
    
    
# =============================================================================
# #NF
# =============================================================================
    if compute_baseline:
        ypred2 = X_test.iloc[:, 3]
        label_cm2 = confusion_matrix(y_test['NF'], ypred2)
        print('Baseline NF')
        print(label_cm2)
        print()
    print('NF')
    print(label_cm[3])
    print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[3], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target NF')
        ax.xaxis.set_ticklabels(['NF NO', 'NF YES'])
        ax.yaxis.set_ticklabels(['NF NO', 'NF YES'])
        # plt.savefig('figure2806/cmMatrixNFW6.pdf', bbox_inches='tight')
        plt.show()

    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base3 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA NF:')
        print("%.3f" % base3)
        print()
    #predicted
    tn, fp, fn, tp = label_cm[3].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba3 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba3)
    print()

    # print baseline NV
    # idx_nv = y_test.columns.get_loc('NV')
    # ypred2 = y_pred[:, idx_nv]
    # label_cm2 = confusion_matrix(y_test['NV'], ypred2)
    # print('Baseline NV')
    # print(label_cm2)
    # print()
    
# =============================================================================
# #NV
# =============================================================================
    if compute_baseline:
        ypred2 = X_test.iloc[:, 4]
        label_cm2 = confusion_matrix(y_test['NV'], ypred2)
        print('Baseline NV')
        print(label_cm2)
        print()
    print('NV')
    print(label_cm[4])
    print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[4], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target NV')
        ax.xaxis.set_ticklabels(['NV NO', 'NV YES'])
        ax.yaxis.set_ticklabels(['NV NO', 'NV YES'])
        # plt.savefig('figure2806/cmMatrixNVW6.pdf', bbox_inches='tight')
        plt.show()

    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base4 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA NV:')
        print("%.3f" % base4)
        print()
    #predicted
    tn, fp, fn, tp = label_cm[4].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba4 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba4)
    print()

    # print baseline SHUTTER
    # idx_shutter = y_test.columns.get_loc('SHUTTER')
    # ypred2 = y_pred[:, idx_shutter]
    # label_cm2 = confusion_matrix(y_test['SHUTTER'], ypred2)
    # print('Baseline SHUTTER')
    # print(label_cm2)
    # print()
    
# =============================================================================
# #SHUTTER
# =============================================================================
    if compute_baseline:
        ypred2 = X_test.iloc[:, 5]
        label_cm2 = confusion_matrix(y_test['SHUTTER'], ypred2)
        print('Baseline SHUTTER')
        print(label_cm2)
        print()
    print('SHUTTER')
    print(label_cm[5])
    print()

    if enable_plots:
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(label_cm[5], annot=True, fmt=".1f", ax=ax)
        ax.set_title('Confusion Matrix Target SHUTTER')
        ax.xaxis.set_ticklabels(['SHUTTER NO', 'SHUTTER YES'])
        ax.yaxis.set_ticklabels(['SHUTTER NO', 'SHUTTER YES'])
        # plt.savefig('figure2806/cmMatrixSHUTTERW6.pdf', bbox_inches='tight')
        plt.show()

    #baseline
    if label_cm2 is not None:
        tn, fp, fn, tp = label_cm2.ravel()
        # Calculate sensitivity (true positive rate) for each class
        sensitivity_class_0 = tn / (tn + fp)
        sensitivity_class_1 = tp / (fn + tp)
        # Calculate balanced accuracy
        base5 = (sensitivity_class_0 + sensitivity_class_1) / 2
        print('Baseline BA SHUTTER:')
        print("%.3f" % base5)
        print()
    #predicted
    tn, fp, fn, tp = label_cm[5].ravel()
    # Calculate sensitivity (true positive rate) for each class
    sensitivity_class_0 = tn / (tn + fp)
    sensitivity_class_1 = tp / (fn + tp)
    # Calculate balanced accuracy
    ba5 = (sensitivity_class_0 + sensitivity_class_1) / 2
    print("%.3f" % ba5)
    print()

  
    return label_cm, ba0, ba1, ba2, ba3, ba4, ba5, base0, base1, base2, base3, base4, base5, y_pred_all