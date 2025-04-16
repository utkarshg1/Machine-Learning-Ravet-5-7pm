import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def preprocess_data(X: pd.DataFrame) -> tuple:
    # Seperate cat and con features
    cat = list(X.columns[X.dtypes == "object"])
    con = list(X.columns[X.dtypes != "object"])

    # Get the preprocessor as per requirement
    if not cat:
        pre = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        ).set_output(transform="pandas")
    else:
        num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        cat_pipe = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        )
        pre = ColumnTransformer(
            [("num", num_pipe, con), ("cat", cat_pipe, cat)]
        ).set_output(transform="pandas")

    # Fit the preprocessor
    X_pre = pre.fit_transform(X)

    # Return the values of X_pre and pre
    return X_pre, pre


def evaluate_single_model(model, xtrain, ytrain, xtest, ytest):
    # Fit the model on train data
    model.fit(xtrain, ytrain)

    # Predict results for train and test
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)

    # Get the f1_macro for train and test
    f1_train = f1_score(ytrain, ypred_train, average="macro")
    f1_test = f1_score(ytest, ypred_test, average="macro")

    # Cross validate results
    scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro", n_jobs=-1)
    f1_cv = scores.mean()

    # Return the dictionary of above
    return {
        "name": type(model).__name__,
        "model": model,
        "f1_train": f1_train,
        "f1_test": f1_test,
        "f1_cv": f1_cv,
    }

def algo_evaluation(models: list, xtrain, ytrain, xtest, ytest):
    res = []
    for model in models:
        r = evaluate_single_model(model, xtrain, ytrain, xtest, ytest)
        res.append(r)
        print(r)
        print("="*100 + "\n")
    
    # Convert results to dataframe
    res_df = pd.DataFrame(res)
    sort_df = res_df.sort_values(by="f1_cv", ascending=False).reset_index(drop=True).round(4)
    best_model = sort_df.loc[0, "model"]
    return sort_df, best_model