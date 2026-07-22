import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

print("Libraries Imported Successfully!")

columns = [

    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"

]


df = pd.read_csv(

    "dataset/adult.data",

    names=columns,

    skipinitialspace=True

)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())
df.replace("?", np.nan, inplace=True)


df.dropna(inplace=True)

print("\nAfter Cleaning:")
print(df.shape)


X = df.drop(

    "income",

    axis=1

)


y = df["income"]


y = y.map({

    "<=50K":0,

    ">50K":1,

    "<=50K.":0,

    ">50K.":1

})

print("\nTarget Distribution:")

print(y.value_counts())


# TRAIN


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

print("\nTrain Test Split Completed!")

# PREPROCESSING


categorical_columns = X_train.select_dtypes(

    include=["object"]

).columns


numerical_columns = X_train.select_dtypes(

    exclude=["object"]

).columns

preprocessor = ColumnTransformer(

    transformers=[


        (
            "num",

            StandardScaler(),

            numerical_columns

        ),


        (
            "cat",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            ),

            categorical_columns

        )

    ]

)


# MODELS 

models = {


    "Logistic Regression":

        LogisticRegression(

            max_iter=1000

        ),



    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42

        ),



    "KNN":

        KNeighborsClassifier(

            n_neighbors=5

        ),



    "Naive Bayes":

        GaussianNB(),



    "Random Forest":

        RandomForestClassifier(

            n_estimators=100,

            random_state=42

        )

}

os.makedirs(

    "models",

    exist_ok=True

)

results = []

for name, model in models.items():
    print("\nTraining:", name)

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor

            ),

            (

                "model",

                model

            )

        ]

    )

    pipeline.fit(

        X_train,

        y_train

    )

    prediction = pipeline.predict(

        X_test

    )

    accuracy = accuracy_score(

        y_test,

        prediction

    )

    precision = precision_score(

        y_test,

        prediction

    )

    recall = recall_score(

        y_test,

        prediction

    )

    f1 = f1_score(

        y_test,

        prediction

    )

    auc = roc_auc_score(

        y_test,

        prediction

    )

    mcc = matthews_corrcoef(

        y_test,

        prediction

    )

    print("Accuracy :", accuracy)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("F1 Score :", f1)

    print("AUC      :", auc)

    print("MCC      :", mcc)

    results.append([


        name,

        accuracy,

        precision,

        recall,

        f1,

        auc,

        mcc


    ])

    file_name = (

        name.lower()

        .replace(" ","_")

        + ".pkl"

    )



    joblib.dump(

        pipeline,

        "models/" + file_name

    )

print("\nAll Models Trained Successfully!")

results_df = pd.DataFrame(

    results,

    columns=[

        "ML Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "AUC",

        "MCC"

    ]

)

print("\nMODEL COMPARISON")

print(results_df)

results_df.to_csv(

    "model_results.csv",

    index=False

)

print("\nmodel_results.csv Created Successfully!")

X_test.to_csv(

    "test_data.csv",
    index=False

)
print("\nTest Data Saved Successfully!")