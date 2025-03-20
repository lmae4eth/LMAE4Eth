import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_data():
    np.random.seed(42)
    X = np.random.randn(2000, 30)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] * X[:, 3] + np.random.randn(2000) * 0.1) > 0
    return X, y.astype(int)


class CustomTransformer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)


def create_keras_model(learning_rate=0.001, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(15,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def build_feature_union():
    return FeatureUnion([
        ('pca', PCA(n_components=10)),
        ('selectkbest', SelectKBest(f_classif, k=10)),
        ('polynomial', PolynomialFeatures(degree=2, include_bias=False))
    ])


X, y = generate_synthetic_data()
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp)

preprocessor = Pipeline([
    ('feature_union', build_feature_union()),
    ('custom_transform', CustomTransformer())
])

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}

svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

nn_params = {
    'batch_size': [32, 64],
    'epochs': [50],
    'model__learning_rate': [0.001, 0.0001],
    'model__dropout_rate': [0.2, 0.3]
}

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svm = SVC(probability=True)
keras_model = KerasClassifier(build_fn=create_keras_model, verbose=0)

pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

pipeline_gb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', gb)
])

pipeline_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', svm)
])

pipeline_nn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', keras_model)
])

grid_rf = GridSearchCV(pipeline_rf, param_grid={'classifier__' + k: v for k, v in rf_params.items()}, cv=3)
grid_gb = GridSearchCV(pipeline_gb, param_grid={'classifier__' + k: v for k, v in gb_params.items()}, cv=3)
grid_svm = GridSearchCV(pipeline_svm, param_grid={'classifier__' + k: v for k, v in svm_params.items()}, cv=3)
grid_nn = GridSearchCV(pipeline_nn, param_grid=nn_params, cv=2)

ensemble = VotingClassifier(
    estimators=[
        ('rf', grid_rf),
        ('gb', grid_gb),
        ('svm', grid_svm),
        ('nn', grid_nn)
    ],
    voting='soft'
)

skf = StratifiedKFold(n_splits=5)
X_preprocessed = preprocessor.fit_transform(X_train, y_train)

final_model = Sequential()
final_model.add(Dense(128, activation='relu', input_shape=(X_preprocessed.shape[1],)))
final_model.add(Dropout(0.3))
final_model.add(Dense(64, activation='relu'))
final_model.add(Dropout(0.3))
final_model.add(Dense(1, activation='sigmoid'))
final_model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = final_model.fit(X_preprocessed, y_train, epochs=100, batch_size=32,
                          validation_split=0.2, callbacks=[early_stop], verbose=0)

X_test_preprocessed = preprocessor.transform(X_test)
y_probs = final_model.predict(X_test_preprocessed).flatten()
y_pred = (y_probs > 0.5).astype(int)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History')
plt.legend()

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

feature_importances = grid_rf.best_estimator_.named_steps['classifier'].feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title('Feature Importances')
plt.show()

roc_data = []
for model in [grid_rf, grid_gb, grid_svm]:
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_data.append(roc_auc_score(y_test, y_prob))

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
for i, model_name in enumerate(['Random Forest', 'Gradient Boosting', 'SVM']):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_data[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()