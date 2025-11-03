import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 分类器
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False


def load_data(path):
    df = pd.read_csv(path)
    return df['func'].tolist(), df['label'].tolist()

def get_classifier(model_name):
    if model_name == "lg":
        return LogisticRegression(max_iter=1000)
    elif model_name == "svm":
        return SVC(kernel="linear", probability=True)
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100)
    elif model_name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    elif model_name == "xgb":
        if xgb_installed:
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        else:
            raise ImportError("XGBoost 未安装，请执行 pip install xgboost")
    else:
        raise ValueError(f"未知模型类型：{model_name}")

def evaluate(y_true, y_pred):
    print(f"[✓] Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"[✓] Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"[✓] Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"[✓] F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"[✓] MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="logistic",
                        choices=["lg", "svm", "rf", "mlp", "xgb"],
                        help="选择分类器类型")
    args = parser.parse_args()

    # 加载数据
    train_texts, train_labels = load_data("devign_train.csv")
    valid_texts, valid_labels = load_data("devign_valid.csv")
    test_texts,  test_labels  = load_data("devign_test.csv")

    # 特征提取
    vectorizer = TfidfVectorizer(max_features=10000, token_pattern=r"(?u)\b\w+\b")
    X_train = vectorizer.fit_transform(train_texts + valid_texts)
    y_train = train_labels + valid_labels
    X_test = vectorizer.transform(test_texts)

    # 初始化分类器
    model = get_classifier(args.model)
    print(f"\n[!] 正在训练模型：{args.model}")
    model.fit(X_train, y_train)

    # 测试与评估
    y_pred = model.predict(X_test)
    print(f"\n[!] 在 Devign 测试集上的结果：")
    evaluate(test_labels, y_pred)

if __name__ == "__main__":
    main()
