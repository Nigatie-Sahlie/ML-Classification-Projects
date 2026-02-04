import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score


#lead the data
def lead_data(path):
    return pd.read_csv(path)


#Calculating the outlier of the datast
def detect_outlier(data, cols):
    if isinstance(cols, str):
        cols = [cols]

    q1 = data[cols].quantile(0.25)
    q3 = data[cols].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - (1.5 * iqr)
    ub = q3 + (1.5 * iqr)

    outlier_mask = (data[cols] < lb) | (data[cols] > ub)
    outlier_rows = data[outlier_mask.any(axis=1)]
    outlier_cols = outlier_mask.any(axis=0)
    outlier_data = outlier_rows.loc[:, outlier_cols]

    return outlier_rows, outlier_data, outlier_mask

def outlier_handler(data, cols, method="remove"):
    outlier_rows, _, mask = detect_outlier(data, cols)

    if method == "remove":
        return data.drop(index=outlier_rows.index)

    if method == "flag":
        data = data.copy()
        data["is_outlier"] = mask.any(axis=1)
        return data


#Graphs and Plots
def sub_plot(data1, data2, nrow, ncol, x, y, collection):
    # 1. Create the grid
    fig, axs = plt.subplots(nrow, ncol, figsize=(x, y))
    fig.suptitle("Fraud vs Legitimate Transactions per Feature", fontsize=16)
    
    # 2. Flatten the axes so we can use a single index (i)
    axs_flat = axs.flatten()

    for i, name in enumerate(collection):
        # 3. Access specific feature from dataframes
        sns.histplot(data1[name], ax=axs_flat[2*i], kde=True, color='orange', label='Fraud', alpha=0.5)
        sns.histplot(data2[name], ax=axs_flat[(2*i)+1], kde=True, color="blue", label='Legitimate', alpha=0.5)
        
        # 4. Set titles and labels for the specific subplot
        axs_flat[2*i].set_title(f'{name} of Fraud')
        axs_flat[(2*i)+1].set_title(f"{name} of Legitmate")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def sub_plot1(data, nrow, ncol, x, y, collections):

    fig, axs = plt.subplots(nrow, ncol, figsize=(x, y))
    fig.suptitle("Data Distribution using Histogram", fontsize=16)
    
    # 2. Flatten the axes so we can use a single index (i)
    axs_flat = axs.flatten()

    for i, name in enumerate(collections):
        sns.histplot(data[name], ax=axs_flat[3*i], kde=True, color='orange', alpha=1, label="Hist")
        sns.boxenplot(x=data[name], ax=axs_flat[(3*i)+1], color="green")
        sns.violinplot(x=data[name], ax=axs_flat[(3*i)+2])
        axs_flat[(3*i)].set_title(f'Histogram of {name}')
        axs_flat[(3*i)+1].set_title(f'Boxenplot of {name}')
        axs_flat[(3*i)+2].set_title(f'Violinplot of {name}')

    plt.tight_layout()

def box_plot(data, nrow, ncol, x, y, collections):
    fig, axis = plt.subplots(nrow, ncol, figsize=(x, y))
    ax=axis.flatten()

    fig.suptitle("Box plot of most outliered features.")

    for i, name in enumerate(collections):
        sns.boxplot(x=data[name], ax=ax[i])
        ax[i].set_title(f'Boxplot of {name}')

    plt.tight_layout()

def accuracy_recall(predicted, actual, dataset = "test"):
    from sklearn.metrics import precision_recall_curve, average_precision_score

    predicted = np.asarray(predicted).ravel()
    actual = np.asarray(actual).ravel()
    if predicted.shape[0] != actual.shape[0]:
        raise ValueError("predicted and actual must have the same length")

    precision, recall, _ = precision_recall_curve(actual, predicted)

    average_precision = average_precision_score(actual, predicted)

    # Plot PR curve
    plt.figure(figsize=(8, 4))
    plt.plot(recall, precision, marker='.', label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve of {dataset} dataset')
    plt.legend()
    plt.grid()
    plt.show()

#confuion matrix

def confusionM_RP_plot(actual, predicted, model ="__", dataset="__"):
    from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, precision_recall_curve, average_precision_score
    print(f"total of obsercation: {len(actual)}")
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    print(f"accuracy: {accuracy * 100:.1f}%")
    print(f"precision: {precision* 100:.1f}%")
    print(f"recall: {recall * 100:.1f}%")

    precision, recall, _ = precision_recall_curve(actual, predicted)
    av_precision_score = average_precision_score(actual, predicted)

    fig, axis = plt.subplots(1, 2, sharey=False, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix and Precision-Recall Curve for {model} model and {dataset} data.", fontsize=20)
    conf_ax, rp_ax = axis
    concuion_matrix = confusion_matrix(actual, predicted)
    sns.heatmap(
        concuion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud'],
        ax=conf_ax
    )
    conf_ax.set_xlabel("Predicted")
    conf_ax.set_ylabel("Actual")
    conf_ax.set_title("Confusion Matrix", color='red')

    rp_ax.plot(recall, precision, marker='.', label=f'AP = {av_precision_score:.2f}')
    rp_ax.set_xlabel('Recall')
    rp_ax.set_ylabel('Precision')
    rp_ax.set_title("Precision-Recall Curve", color='orange')
    rp_ax.legend()
    rp_ax.grid()
    plt.tight_layout()
    plt.show()


def accuracy_recall_on_ax(predicted, actual, ax, label=None, title=None, color=None, selector=None):
    predicted = np.asarray(predicted).ravel()
    actual = np.asarray(actual).ravel()
    
    match selector:
        case "PR":
            precision, recall, _ = precision_recall_curve(actual, predicted)
            average_precision = average_precision_score(actual, predicted)

            ax.plot(recall, precision, marker='.', label=f"{label} (AP={average_precision:.2f})", color=color)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(title or 'Precision-Recall Curve')
            ax.grid(True)
            ax.legend()

        case "ROC":
            fpr, tpr, thresholds = roc_curve(actual, predicted)
            roc_auc = roc_auc_score(actual, predicted)

            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=color)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Baseline
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate (Recall)")
            ax.set_title(title or "ROC Curve")
            ax.grid(True)
            ax.legend()
            
        case _:
            print("Invalid selector. Please use 'PR' or 'ROC'.")

   
