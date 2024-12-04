import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import shap
import yaml
import os

class FeatureEvaluator:
    def __init__(self, df: pd.DataFrame, target_col: str = 'home_team_won'):
        """
        Initialize the feature evaluator.
        
        Parameters:
            df: DataFrame with features and target
            target_col: Name of the target column (default: 'home_team_won')
        """
        self.df = df.copy()
        
        # Create binary target if not exists
        if target_col not in self.df.columns:
            self.df[target_col] = (self.df['home_team_score'] > self.df['away_team_score']).astype(int)
        
        self.target_col = target_col
        self.feature_cols = [col for col in df.columns if col not in [
            target_col, 'date', 'home_team', 'away_team', 
            'home_team_score', 'away_team_score'
        ]]
        
        # Initialize containers for results
        self.correlation_scores = {}
        self.mutual_info_scores = {}
        self.feature_importances = {}
        self.shap_values = None
        self.pca = None
        self.pca_components = None
        self.explained_variance_ratio = None

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation with target and between features"""
        print("Calculating correlations...")
        
        # Calculate correlations
        correlations = self.df[self.feature_cols + [self.target_col]].corr()
        
        # Store target correlations
        self.correlation_scores = correlations[self.target_col].sort_values(ascending=False)
        
        return correlations

    def calculate_mutual_information(self) -> Dict[str, float]:
        """Calculate mutual information scores"""
        print("Calculating mutual information scores...")
        
        for feature in self.feature_cols:
            mi_score = mutual_info_score(
                self.df[feature].fillna(self.df[feature].mean()),
                self.df[self.target_col]
            )
            self.mutual_info_scores[feature] = mi_score
        
        return dict(sorted(self.mutual_info_scores.items(), key=lambda x: x[1], reverse=True))

    def train_random_forest(self) -> Tuple[RandomForestClassifier, float]:
        """Train a random forest and get feature importances"""
        print("Training Random Forest for feature importance...")
        
        X_scaled, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        self.feature_importances = dict(zip(
            self.feature_cols,
            rf.feature_importances_
        ))
        
        # Calculate model performance
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        prediction_accuracy = rf.score(X_test, y_test)
        
        return rf, auc_score, prediction_accuracy

    def calculate_shap_values(self, rf_model: RandomForestClassifier) -> None:
        """Calculate SHAP values for feature importance"""
        print("Calculating SHAP values...")
        
        X_scaled, _ = self.prepare_data()
        explainer = shap.TreeExplainer(rf_model)
        self.shap_values = explainer.shap_values(X_scaled)

    def perform_pca(self, n_components: int = None) -> None:
        """Perform PCA on the features and plot explained variance"""
        print("Performing PCA...")
        X_scaled, _ = self.prepare_data()
        if n_components is None:
            n_components = min(len(self.feature_cols), 10)  # Default to min(10, num_features)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.pca_components = pd.DataFrame(self.pca.components_, columns=self.feature_cols)
        
        # Plot cumulative explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.explained_variance_ratio * 100), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance (%)')
        plt.title('Explained Variance by Number of Principal Components')
        plt.grid(True)
        plt.show()
        
        # Plot the components heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            self.pca_components,
            cmap='viridis',
            annot=True,
            fmt=".2f",
            annot_kws={"size": 5},
            xticklabels=self.feature_cols,
            yticklabels=[f'PC{i+1}' for i in range(self.pca_components.shape[0])]
        )
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        plt.title('PCA Component Loadings')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=(15, 12))
        correlations = self.calculate_correlations()
        
        sns.heatmap(
            correlations,
            cmap='RdBu',
            center=0,
            annot=True,
            fmt='.2f',
            square=True,
            cbar_kws={"shrink": .5},
            annot_kws={"size": 5}  # Set smaller text size for annotations
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self) -> None:
        """Plot different feature importance metrics"""
        # Prepare data
        importance_df = pd.DataFrame({
            'Correlation': self.correlation_scores.abs(),
            'Mutual Information': pd.Series(self.mutual_info_scores),
            'Random Forest': pd.Series(self.feature_importances)
        })
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 20))
        
        # Correlation plot
        importance_df['Correlation'].sort_values().plot(
            kind='barh',
            ax=axes[0],
            title='Feature Importance by Correlation'
        )
        
        # Mutual Information plot
        importance_df['Mutual Information'].sort_values().plot(
            kind='barh',
            ax=axes[1],
            title='Feature Importance by Mutual Information'
        )
        
        # Random Forest importance plot
        importance_df['Random Forest'].sort_values().plot(
            kind='barh',
            ax=axes[2],
            title='Feature Importance by Random Forest'
        )
        
        plt.tight_layout()
        plt.show()

    def plot_shap_summary(self) -> None:
        """Plot SHAP summary plot"""
        if self.shap_values is not None:
            shap.summary_plot(
                self.shap_values,
                self.df[self.feature_cols],
                plot_type="bar"
            )

    def identify_top_features(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify top features using multiple metrics
        
        Parameters:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature rankings across different metrics
        """
        rankings = pd.DataFrame({
            'Correlation': self.correlation_scores.abs().rank(ascending=False),
            'Mutual Information': pd.Series(self.mutual_info_scores).rank(ascending=False),
            'Random Forest': pd.Series(self.feature_importances).rank(ascending=False)
        })
        
        # Calculate average ranking
        rankings['Average Rank'] = rankings.mean(axis=1)
        rankings = rankings.sort_values('Average Rank')
        
        return rankings.head(top_n)

def main():
    # Load files
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    with open(BASE_DIRECTORY + "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Load your feature-engineered data
    df = pd.read_parquet(config["processed_data_with_features"])

    # Initialize evaluator
    evaluator = FeatureEvaluator(df.drop(columns=["gender"]))
    
    # Run analysis
    evaluator.calculate_correlations()
    evaluator.calculate_mutual_information()
    rf_model, auc_score, prediction_accuracy = evaluator.train_random_forest()
    # evaluator.calculate_shap_values(rf_model)
    evaluator.perform_pca()
    
    # Generate visualizations
    evaluator.plot_correlation_matrix()
    evaluator.plot_feature_importances()
    # evaluator.plot_shap_summary()
    
    # Get top features
    top_features = evaluator.identify_top_features(top_n=10)
    print("\nTop 10 Most Important Features:")
    print(top_features)
    
    print(f"\nRandom Forest Model AUC Score: {auc_score:.3f}")
    print(f"\nRandom Forest Model Accuracy: {prediction_accuracy*100:.3f}%")

    print(f"Features in the dataset: {df.columns}")

if __name__ == "__main__":
    main()
