import ast
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE


def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def get_embedding(X_test, X_train, mode):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[f'text_{mode}'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[f'text_{mode}'])

    # Converting sparse matrices to dense
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    return X_test_dense, X_train_dense

def get_lm_embedding(data, embedder):
    train_embeddings = data[embedder + '_embedding'].apply(ast.literal_eval).tolist()
    train_embeddings = np.array(train_embeddings)
    return train_embeddings

def oversampling_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Train data distribution After Oversampling: {y_train_resampled.value_counts()}")

    return X_train_resampled, y_train_resampled

def get_splitted_data(train_df, test_df, mode, embedding, oversampling):
    solution_col = 'label'
    X_train, y_train = train_df.drop(solution_col, axis=1), train_df[solution_col]
    X_test, y_test = test_df.drop(solution_col, axis=1), test_df[solution_col]

    if embedding in ["bert", "llama", "gpt"]:
        X_test_embedding = get_lm_embedding(X_test, embedding)
        X_train_embedding = get_lm_embedding(X_train, embedding)
    else:
        X_test_embedding, X_train_embedding = get_embedding(X_test, X_train, mode)

    if oversampling:
        X_train_embedding, y_train = oversampling_data(X_train_embedding, y_train)

    return X_train_embedding, X_test_embedding, y_train, y_test

if __name__ == "__main__":
    eval_dataset = ""
    model_path = ""

    data = pd.read_csv()

    model = load_model(model_path)
    y_pred = model.predict(X_test)