import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, confusion_matrix


#lead the data
def lead_data(path):
    return pd.read_csv(path)

#the relation of the features using correlation and scatter plot
def correlation_plot(data):
    cor = data.corr()
    plt.figure(figsize=(20, 10))
    sns.heatmap(cor, annot=True, fmt=".1f", cmap='coolwarm')
    plt.title("Correlation Heatmap of Features")
    plt.show()

#for scatter plot
def scatter_plot(data, x1, y1, x2, y2, hue=None):
    fig, axis = plt.subplots(1, 2, sharex=False, figsize=(20, 6))
    first, second = axis
    sns.scatterplot(data=data, x=x1, y=y1, hue=hue, ax=first)
    sns.scatterplot(data=data, x=x2, y=y2, hue=hue, ax=second)
    first.set_title(f'Scatter Plot of {y1} vs {x1}')
    second.set_title(f'Scatter Plot of {y2} vs {x2}')
    plt.show()

#The distribution of the dataset
def data_dist(data, columns):
    fig, axis =plt.subplots(5, 6, figsize=(30, 15))
    ax = axis.flatten()
    hue_col = "Class" if "Class" in data.columns else None
    for i, name in enumerate(columns):
        sns.histplot(data=data, x=name, ax=ax[i], kde=True, hue=hue_col)
        ax[i].set_title(f'Histogram of {name}')
    plt.tight_layout()
    plt.show()

#for ploting in pair
def plot_in_pair(data1, data2):
    figure, axis = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
    figure.suptitle("Distribution of Fraud and Legitimate Transactions for {} features".format(data1.name))
    sns.histplot(data=data1, ax=axis[0], kde=True, color='blue', label='Fraud')
    sns.histplot(data=data2, ax=axis[1], kde=True, color='red', label='Legitimate')
    axis[0].legend()
    axis[1].legend()
    plt.show()

#distribution of data for both histogram, boxplot and violine plot
def sub_plots(data, collections, nrow=3):

    fig, axs = plt.subplots(nrow, len(collections), figsize=(len(collections)*5, len(collections)*(5/3)))
    fig.suptitle("Data Distribution", fontsize=16)
    
    # 2. Flatten the axes so we can use a single index (i)
    axs_flat = axs.flatten()

    for i, name in enumerate(collections):
        sns.histplot(data[name], ax=axs_flat[i], kde=True, color='orange', alpha=1, label="Hist")
        sns.boxenplot(x=data[name], ax=axs_flat[len(collections)+i], color="green")
        sns.violinplot(x=data[name], ax=axs_flat[(2*len(collections))+i])
        axs_flat[i].set_title(f'Histogram of {name}')
        axs_flat[len(collections)+i].set_title(f'Boxenplot of {name}')
        axs_flat[(2*len(collections))+i].set_title(f'Violinplot of {name}')
    plt.tight_layout()

#Calculating the outlier of the datast
def detect_outlier(data, cols):
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

#Scaler
def scaler(train, test, type=None):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    match type:
        case "Robust":
            scaler = RobustScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
        case "standard":
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
        case "_":
            print("No scaling applied")
    return train_scaled, test_scaled

#some pipeline for linear models
def train_models(x_train, y_train, which_model=None, params=None):
    match which_model:
        case "Log_regression":
            model = LogisticRegression()
            return model.fit(x_train, y_train)
        case "KNN":
            model = KNeighborsClassifier()
            return model.fit(x_train, y_train)
        case "SVM":
            model = SVC()
            return model.fit(x_train, y_train)
        case "DecisionTree" | "Decision_Tree":
            model = DecisionTreeClassifier()
            return model.fit(x_train, y_train)
        case "RandomForest":
            model = RandomForestClassifier()
            return model.fit(x_train, y_train)
        case "catBoost":
            model = CatBoostClassifier(verbose=False)
            return model.fit(x_train, y_train)
        case "XGBoost":
            model = XGBClassifier()
            return model.fit(x_train, y_train)
        case _:
            print(" please Specify a valid model")

#make prediction
def predict_model(model, features):
    return model.predict(features)

#confuion matrix
def confusionM_RP_plot(train_actual, train_predicted, test_actual, test_predicted, model ="__"):
    # print(f"total of obsercation: {len(train_actual)}")
    # accuracy = accuracy_score(train_actual, train_predicted)
    # precision = precision_score(train_actual, train_predicted)
    # recall = recall_score(train_actual, train_predicted)
    # print(f"accuracy: {accuracy * 100:.1f}%")
    # print(f"precision: {precision* 100:.1f}%")
    # print(f"recall: {recall * 100:.1f}%")

    #for train data
    train_precision, train_recall, _ = precision_recall_curve(train_actual, train_predicted)
    train_av_precision_score = average_precision_score(train_actual, train_predicted)

    fig, axis = plt.subplots(2, 2, sharey=False, figsize=(20, 10))
    fig.suptitle(f"Confusion Matrix and Precision-Recall Curve for {model} model with train and test datasets", fontsize=20)
  
    conf_train_ax, rp_train_ax, conf_test_ax, rp_test_ax = axis.flatten()
    train_confusion_matrix = confusion_matrix(train_actual, train_predicted)
    sns.heatmap(
        train_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud'],
        ax=conf_train_ax
    )
    conf_train_ax.set_xlabel("Predicted")
    conf_train_ax.set_ylabel("Actual")
    conf_train_ax.set_title("Confusion Matrix", color='red')

    rp_train_ax.plot(train_recall, train_precision, marker='.', label=f'AP = {train_av_precision_score:.2f}')
    rp_train_ax.set_xlabel('Recall')
    rp_train_ax.set_ylabel('Precision')
    rp_train_ax.set_title("Precision-Recall Curve", color='orange')
    rp_train_ax.legend()
    rp_train_ax.grid()

    #for test dataset
    test_precision, test_recall, _ = precision_recall_curve(test_actual, test_predicted)
    test_av_precision_score = average_precision_score(test_actual, test_predicted)

    test_confusion_matrix = confusion_matrix(test_actual, test_predicted)
    sns.heatmap(
        test_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud'],
        ax=conf_test_ax
    )
    conf_test_ax.set_xlabel("Predicted")
    conf_test_ax.set_ylabel("Actual")
    conf_test_ax.set_title("Confusion Matrix", color='red')
    rp_test_ax.plot(test_recall, test_precision, marker='.', label=f'AP = {test_av_precision_score:.2f}')
    rp_test_ax.set_xlabel('Recall')
    rp_test_ax.set_ylabel('Precision')
    rp_test_ax.set_title("Precision-Recall Curve", color='orange')
    rp_test_ax.legend()
    rp_test_ax.grid()
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()

#accuracy and recall on train and test data
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

   
