# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import openpyxl  # For Excel support
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("⚠️ Excel support not available. Install with: pip install openpyxl")

warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 8
sns.set_style("whitegrid")

class HeartDiseasePredictor:
    """
    A comprehensive class for heart disease prediction using ensemble methods
    """
    
    def __init__(self, data_path='heart.csv'):
        """
        Initialize the predictor with dataset path
        
        Args:
            data_path (str): Path to the heart disease dataset
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.label_encoders = {}  # Store encoders for later use
        
        print("🚀 Heart Disease Prediction System Initialized")
        print("=" * 60)
    
    def load_data(self):
        """
        Load the heart disease dataset and perform initial inspection
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        print("📥 LOADING DATASET")
        print("-" * 30)
        
        try:
            # Try to load as CSV first
            if self.data_path.endswith('.csv'):
                self.df = pd.read_csv(self.data_path)
            # If it's Excel file and Excel support is available
            elif self.data_path.endswith(('.xlsx', '.xls')) and EXCEL_AVAILABLE:
                self.df = pd.read_excel(self.data_path)
            else:
                # Try CSV anyway
                self.df = pd.read_csv(self.data_path)
                
            print(f"✅ Dataset loaded successfully from '{self.data_path}'")
            print(f"📊 Dataset shape: {self.df.shape} (rows, columns)")
            print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Display basic information
            print(f"\n📋 Column names:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            # Show first few rows
            print(f"\n🔍 First 5 rows preview:")
            print(self.df.head())
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File '{self.data_path}' not found!")
            print("Please make sure the file exists in the correct directory.")
            print("You can try these steps:")
            print("  1. Check if the file path is correct")
            print("  2. Make sure the file exists")
            print("  3. Try using full path instead of relative path")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def perform_eda(self):
        """
        Perform comprehensive Exploratory Data Analysis
        """
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
        
        print("\n📈 EXPLORATORY DATA ANALYSIS")
        print("-" * 40)
        
        # Basic dataset information
        print("📊 Dataset Overview:")
        print(f"  • Total samples: {len(self.df):,}")
        print(f"  • Total features: {len(self.df.columns)}")
        print(f"  • Data types:")
        
        # Show data types
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"    - {dtype}: {count} columns")
        
        # Check for missing values
        print(f"\n❓ Missing Values Analysis:")
        missing_values = self.df.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing == 0:
            print("  ✅ No missing values found - Clean dataset!")
        else:
            print(f"  ⚠️  Total missing values: {total_missing}")
            print(f"  📋 Missing values by column:")
            for col, missing in missing_values[missing_values > 0].items():
                percentage = (missing / len(self.df)) * 100
                print(f"    - {col}: {missing} ({percentage:.1f}%)")
        
        # Analyze target variable
        target_col = None
        possible_targets = ['HeartDisease', 'target', 'heart_disease', 'output']
        
        for col in possible_targets:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\n🎯 Target Variable Analysis ('{target_col}'):")
            target_counts = self.df[target_col].value_counts().sort_index()
            
            for value, count in target_counts.items():
                percentage = (count / len(self.df)) * 100
                label = "No Heart Disease" if value == 0 else "Heart Disease"
                print(f"  • {label} ({value}): {count:,} samples ({percentage:.1f}%)")
            
            # Check class balance
            minority_class = target_counts.min()
            majority_class = target_counts.max()
            balance_ratio = minority_class / majority_class
            
            print(f"  📊 Class balance ratio: {balance_ratio:.2f}")
            if balance_ratio < 0.7:
                print("  ⚠️  Warning: Significant class imbalance detected!")
            else:
                print("  ✅ Classes are reasonably balanced")
        else:
            print("\n⚠️ Target variable not found. Please check column names.")
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        print(self.df.describe())
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for data understanding
        """
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
        
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 35)
        
        # Find target column
        target_col = None
        possible_targets = ['HeartDisease', 'target', 'heart_disease', 'output']
        
        for col in possible_targets:
            if col in self.df.columns:
                target_col = col
                break
        
        if not target_col:
            print("❌ Target variable not found. Cannot create target-based visualizations.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main visualization figure
        fig = plt.figure(figsize=(16, 13))
        fig.suptitle('Heart Disease Dataset - Comprehensive Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Target variable distribution
        plt.subplot(3, 4, 1)
        target_counts = self.df[target_col].value_counts()
        colors = ['lightblue', 'lightcoral']
        bars = plt.bar(['No Disease', 'Heart Disease'], target_counts.values, color=colors)
        plt.title('Target Variable Distribution', fontweight='bold')
        plt.ylabel('Number of Patients')
        
        # Add value labels on bars
        for bar, value in zip(bars, target_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Age distribution (if Age column exists)
        if 'Age' in self.df.columns:
            plt.subplot(3, 4, 2)
            plt.hist(self.df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Age Distribution', fontweight='bold')
            plt.xlabel('Age (years)')
            plt.ylabel('Frequency')
            plt.axvline(self.df['Age'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["Age"].mean():.1f}')
            plt.legend()
        
        # 3. Cholesterol distribution (if Cholesterol column exists)
        if 'Cholesterol' in self.df.columns:
            plt.subplot(3, 4, 3)
            # Filter out zero values for cholesterol (likely missing data)
            data_filtered = self.df[self.df['Cholesterol'] > 0]['Cholesterol']
            plt.hist(data_filtered, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Cholesterol Distribution', fontweight='bold')
            plt.xlabel('Cholesterol (mg/dl)')
            plt.ylabel('Frequency')
            if len(data_filtered) > 0:
                plt.axvline(data_filtered.mean(), color='red', linestyle='--',
                           label=f'Mean: {data_filtered.mean():.0f}')
                plt.legend()
        
        # 4. Chest Pain Type distribution (if exists)
        chest_pain_cols = ['ChestPainType', 'cp', 'chest_pain']
        chest_pain_col = None
        for col in chest_pain_cols:
            if col in self.df.columns:
                chest_pain_col = col
                break
        
        if chest_pain_col:
            plt.subplot(3, 4, 4)
            chest_pain_counts = self.df[chest_pain_col].value_counts()
            plt.bar(range(len(chest_pain_counts)), chest_pain_counts.values, 
                   color='lightcoral', alpha=0.8)
            plt.title('Chest Pain Type Distribution', fontweight='bold')
            plt.xlabel('Chest Pain Type')
            plt.ylabel('Count')
            plt.xticks(range(len(chest_pain_counts)), chest_pain_counts.index, rotation=45)
        
        # Continue with other plots if columns exist
        plot_idx = 5
        
        # ST_Slope distribution
        st_slope_cols = ['ST_Slope', 'st_slope', 'slope']
        st_slope_col = None
        for col in st_slope_cols:
            if col in self.df.columns:
                st_slope_col = col
                break
        
        if st_slope_col:
            plt.subplot(3, 4, plot_idx)
            st_slope_counts = self.df[st_slope_col].value_counts()
            plt.bar(range(len(st_slope_counts)), st_slope_counts.values, 
                   color='gold', alpha=0.8)
            plt.title('ST Slope Distribution', fontweight='bold')
            plt.xlabel('ST Slope')
            plt.ylabel('Count')
            plt.xticks(range(len(st_slope_counts)), st_slope_counts.index)
            plot_idx += 1
        
        # Exercise Angina distribution
        angina_cols = ['ExerciseAngina', 'exercise_angina', 'angina']
        angina_col = None
        for col in angina_cols:
            if col in self.df.columns:
                angina_col = col
                break
        
        if angina_col:
            plt.subplot(3, 4, plot_idx)
            angina_counts = self.df[angina_col].value_counts()
            plt.pie(angina_counts.values, labels=['No Angina', 'Exercise Angina'], 
                   autopct='%1.1f%%', colors=['lightblue', 'orange'])
            plt.title('Exercise Angina Distribution', fontweight='bold')
            plot_idx += 1
        
        # Age vs Heart Disease
        if 'Age' in self.df.columns:
            plt.subplot(3, 4, plot_idx)
            sns.boxplot(data=self.df, x=target_col, y='Age')
            plt.title('Age Distribution by Heart Disease', fontweight='bold')
            plt.xlabel('Heart Disease (0=No, 1=Yes)')
            plt.ylabel('Age (years)')
            plot_idx += 1
        
        # Cholesterol vs Heart Disease
        if 'Cholesterol' in self.df.columns:
            plt.subplot(3, 4, plot_idx)
            # Filter out zero values
            df_filtered = self.df[self.df['Cholesterol'] > 0]
            if len(df_filtered) > 0:
                sns.boxplot(data=df_filtered, x=target_col, y='Cholesterol')
                plt.title('Cholesterol by Heart Disease', fontweight='bold')
                plt.xlabel('Heart Disease (0=No, 1=Yes)')
                plt.ylabel('Cholesterol (mg/dl)')
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()
        
        # Create correlation heatmap
        print("🔥 Creating correlation heatmap...")
        self._create_correlation_heatmap()
        
        print("✅ All visualizations created successfully!")
    
    def _create_correlation_heatmap(self):
        """
        Create a correlation heatmap for all numerical features
        """
        try:
            # Prepare data for correlation (encode categorical variables)
            df_corr = self.df.copy()
            
            # Encode categorical variables
            categorical_columns = df_corr.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_columns:
                if col in df_corr.columns:
                    le = LabelEncoder()
                    df_corr[col] = le.fit_transform(df_corr[col].astype(str))
            
            # Calculate correlation matrix
            correlation_matrix = df_corr.corr()
            
            # Create the heatmap
            plt.figure(figsize=(10, 8))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       fmt='.2f', 
                       square=True, 
                       linewidths=0.5,
                       cbar_kws={"shrink": .8})
            
            plt.title('Feature Correlation Matrix\n(Categorical variables encoded)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not create correlation heatmap: {str(e)}")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data for machine learning - keeping only 5 features
        
        Args:
            test_size (float): Proportion of dataset for testing
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
        
        print("\n🔧 DATA PREPROCESSING")
        print("-" * 30)
        
        # Create a copy for processing
        self.df_processed = self.df.copy()
        
        # Find target column
        target_col = None
        possible_targets = ['HeartDisease', 'target', 'heart_disease', 'output']
        
        for col in possible_targets:
            if col in self.df_processed.columns:
                target_col = col
                break
        
        if not target_col:
            print("❌ Target variable not found. Please check column names.")
            return
        
        print("📝 Step 1: Selecting only 5 features")
        
        # Define the 5 features to keep - map different possible column names
        feature_mapping = {
            'Age': ['Age', 'age'],
            'ST_Slope': ['ST_Slope', 'st_slope', 'slope'],
            'ExerciseAngina': ['ExerciseAngina', 'exercise_angina', 'angina', 'ExerciseInducedAngina'],
            'ChestPainType': ['ChestPainType', 'cp', 'chest_pain', 'ChestPain'],
            'Cholesterol': ['Cholesterol', 'chol', 'cholesterol', 'Chol']
        }
        
        # Find actual column names in the dataset
        selected_features = {}
        for standard_name, possible_names in feature_mapping.items():
            for name in possible_names:
                if name in self.df_processed.columns:
                    selected_features[standard_name] = name
                    break
        
        print(f"  ✅ Found features:")
        for standard_name, actual_name in selected_features.items():
            print(f"    • {standard_name}: {actual_name}")
        
        missing_features = set(feature_mapping.keys()) - set(selected_features.keys())
        if missing_features:
            print(f"  ⚠️  Missing features: {missing_features}")
            print(f"  Available columns: {list(self.df_processed.columns)}")
        
        # Keep only the selected features and target
        columns_to_keep = list(selected_features.values()) + [target_col]
        self.df_processed = self.df_processed[columns_to_keep]
        
        print(f"  📊 Dataset reduced from {len(self.df.columns)} to {len(self.df_processed.columns)} columns")
        
        print("\n📝 Step 2: Handling categorical variables")
        
        # Get categorical columns
        categorical_columns = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target if it's categorical
        if target_col in categorical_columns:
            categorical_columns.remove(target_col)
        
        # Encode categorical variables
        encoding_info = {}
        for col in categorical_columns:
            le = LabelEncoder()
            original_values = self.df_processed[col].unique()
            self.df_processed[col] = le.fit_transform(self.df_processed[col])
            encoded_values = sorted(self.df_processed[col].unique())
            
            # Store the encoder for later use
            self.label_encoders[col] = le
            
            encoding_info[col] = dict(zip(original_values, encoded_values))
            print(f"  ✅ {col}: {len(original_values)} categories encoded")
        
        # Display encoding information
        if encoding_info:
            print("\n📋 Encoding mapping:")
            for col, mapping in encoding_info.items():
                print(f"  {col}:")
                for original, encoded in mapping.items():
                    print(f"    '{original}' → {encoded}")
        
        print("\n📝 Step 3: Separating features and target")
        
        # Separate features and target
        X = self.df_processed.drop(columns=[target_col])
        y = self.df_processed[target_col]
        
        self.feature_names = X.columns.tolist()
        
        print(f"  • Features (X): {X.shape}")
        print(f"  • Target (y): {y.shape}")
        print(f"  • Feature names: {self.feature_names}")
        
        print("\n📝 Step 4: Train-test split")
        
        # Split the data with stratification to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"  • Training samples: {len(self.X_train):,} ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  • Testing samples: {len(self.X_test):,} ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Check stratification worked
        train_dist = self.y_train.value_counts(normalize=True).sort_index()
        test_dist = self.y_test.value_counts(normalize=True).sort_index()
        
        print(f"  • Training target distribution: {train_dist.values}")
        print(f"  • Testing target distribution: {test_dist.values}")
        
        print("\n📝 Step 5: Feature scaling")
        
        # Scale features using StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames for easier handling
        self.X_train_scaled = pd.DataFrame(X_train_scaled, 
                                          columns=self.feature_names, 
                                          index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, 
                                         columns=self.feature_names, 
                                         index=self.X_test.index)
        
        print(f"  ✅ Features scaled successfully!")
        print(f"  • Original feature range: [{self.X_train.min().min():.2f}, {self.X_train.max().max():.2f}]")
        print(f"  • Scaled feature range: [{self.X_train_scaled.min().min():.2f}, {self.X_train_scaled.max().max():.2f}]")
        
        print(f"\n✅ Data preprocessing completed successfully!")
        print(f"🎯 Using 5 features: {self.feature_names}")
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        if self.X_train_scaled is None:
            print("❌ Data not preprocessed. Please run preprocess_data() first.")
            return
        
        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("-" * 45)
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_to_train['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train the model
                model.fit(self.X_train_scaled, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc_roc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Store model and results
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name} trained successfully!")
                print(f"     • Accuracy: {accuracy:.4f}")
                print(f"     • F1-Score: {f1:.4f}")
                print(f"     • AUC-ROC: {auc_roc:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {str(e)}")
        
        print(f"\n🎉 Model training completed!")
        
    def evaluate_models(self):
        """
        Evaluate and compare all trained models
        """
        if not self.results:
            print("❌ No models trained. Please run train_models() first.")
            return None
        
        print("\n📊 MODEL EVALUATION AND COMPARISON")
        print("-" * 45)
        
        # Create results comparison DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
        results_df = results_df.round(4)
        
        print("📈 Performance Comparison:")
        print(results_df)
        
        # Find best model
        best_model_name = results_df['f1_score'].idxmax()
        best_f1_score = results_df.loc[best_model_name, 'f1_score']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {best_f1_score:.4f}")
        
        # Create performance comparison plots
        self._plot_model_comparison(results_df)
        
        # Create confusion matrices for all models
        self._plot_confusion_matrices()
        
        # Show detailed classification report for best model
        print(f"\n📋 Detailed Classification Report for {best_model_name}:")
        y_pred_best = self.results[best_model_name]['predictions']
        print(classification_report(self.y_test, y_pred_best, 
                                  target_names=['No Disease', 'Heart Disease']))
        
        return best_model_name
    
    def _plot_model_comparison(self, results_df):
        """
        Plot model performance comparison
        """
        try:
            # Set up the plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                ax = axes[i//2, i%2]
                
                # Create bar plot
                bars = ax.bar(results_df.index, results_df[metric], color=color, alpha=0.8)
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Rotate x-axis labels
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, results_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            # Create AUC-ROC comparison
            plt.figure(figsize=(10, 6))
            auc_scores = results_df['auc_roc'].sort_values(ascending=True)
            bars = plt.barh(range(len(auc_scores)), auc_scores.values, color='purple', alpha=0.7)
            plt.yticks(range(len(auc_scores)), auc_scores.index)
            plt.xlabel('AUC-ROC Score')
            plt.title('AUC-ROC Comparison', fontweight='bold', fontsize=14)
            plt.xlim(0, 1)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, auc_scores.values)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Error creating comparison plots: {str(e)}")
    
    def _plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        try:
            n_models = len(self.models)
            cols = 3
            rows = (n_models + cols - 1) // cols  # Ceiling division
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
            
            # Handle case where we have only one row
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif n_models == 1:
                axes = np.array([[axes]])
            
            axes = axes.flatten()
            
            for i, (name, results) in enumerate(self.results.items()):
                ax = axes[i]
                
                # Create confusion matrix
                cm = confusion_matrix(self.y_test, results['predictions'])
                
                # Plot heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['No Disease', 'Heart Disease'],
                           yticklabels=['No Disease', 'Heart Disease'])
                
                ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}', fontweight='bold')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Error creating confusion matrices: {str(e)}")
    
    def predict_new_patient(self, patient_data):
        """
        Predict heart disease for a new patient
        
        Args:
            patient_data (dict): Dictionary with patient features
        
        Returns:
            dict: Predictions from all models
        """
        if not self.models:
            print("❌ No models trained. Please run train_models() first.")
            return None
        
        print("\n🔮 PREDICTING FOR NEW PATIENT")
        print("-" * 35)
        
        try:
            # Convert patient data to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Encode categorical variables using stored encoders
            for col, encoder in self.label_encoders.items():
                if col in patient_df.columns:
                    try:
                        patient_df[col] = encoder.transform([patient_data[col]])
                    except ValueError as e:
                        print(f"⚠️ Warning: Unknown category '{patient_data[col]}' for {col}")
                        # Use the most frequent category as default
                        patient_df[col] = 0
            
            # Ensure all features are present and in the correct order
            for feature in self.feature_names:
                if feature not in patient_df.columns:
                    print(f"⚠️ Warning: Missing feature '{feature}', setting to 0")
                    patient_df[feature] = 0
            
            # Reorder columns to match training data
            patient_df = patient_df[self.feature_names]
            
            # Scale the features
            patient_scaled = self.scaler.transform(patient_df)
            patient_scaled_df = pd.DataFrame(patient_scaled, columns=self.feature_names)
            
            print("👤 Patient Information:")
            for feature, value in patient_data.items():
                print(f"  • {feature}: {value}")
            
            print(f"\n🎯 Predictions:")
            predictions = {}
            
            for name, model in self.models.items():
                try:
                    # Make prediction
                    pred = model.predict(patient_scaled_df)[0]
                    pred_proba = model.predict_proba(patient_scaled_df)[0]
                    
                    predictions[name] = {
                        'prediction': pred,
                        'probability_no_disease': pred_proba[0],
                        'probability_heart_disease': pred_proba[1]
                    }
                    
                    result = "Heart Disease" if pred == 1 else "No Heart Disease"
                    confidence = pred_proba[1] if pred == 1 else pred_proba[0]
                    
                    print(f"  🤖 {name}: {result} (Confidence: {confidence:.1%})")
                    
                except Exception as e:
                    print(f"  ❌ Error with {name}: {str(e)}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def run_complete_analysis(self):
        """
        Run the complete heart disease prediction pipeline
        """
        print("🚀 STARTING COMPLETE HEART DISEASE ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Perform EDA
        self.perform_eda()
        
        # Step 3: Create visualizations
        self.create_visualizations()
        
        # Step 4: Preprocess data
        self.preprocess_data()
        
        if self.X_train_scaled is None:
            print("❌ Data preprocessing failed. Cannot continue.")
            return False
        
        # Step 5: Train models
        self.train_models()
        
        if not self.results:
            print("❌ No models were successfully trained. Cannot continue.")
            return False
        
        # Step 6: Evaluate models
        best_model = self.evaluate_models()
        
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best performing model: {best_model}")
        print(f"🎯 Using 5 features: {self.feature_names}")
        print("=" * 60)
        
        return True

# ========================================
# USAGE EXAMPLE AND TUTORIAL
# ========================================

def main():
    """
    Main function demonstrating how to use the HeartDiseasePredictor class
    """
    print("💝 HEART DISEASE PREDICTION TUTORIAL")
    print("=" * 50)
    
    # AUTO-DETECT DATA FILE
    # Get current directory
    current_dir = os.getcwd()
    print(f"🔍 Looking for data files in: {current_dir}")
    
    # List of possible file names
    possible_files = [
        'heart.csv', 'heart.xlsx', 'heart.xls',
        'heart_disease.csv', 'heart_disease.xlsx', 'heart_disease.xls',
        'heartdisease.csv', 'heartdisease.xlsx', 'heartdisease.xls',
        'data.csv', 'data.xlsx', 'data.xls',
        'heart_data.csv', 'heart_data.xlsx', 'heart_data.xls'
    ]
    
    # Look for files in current directory
    data_path = None
    for filename in possible_files:
        full_path = os.path.join(current_dir, filename)
        if os.path.exists(full_path):
            data_path = full_path
            print(f"🔍 Found data file: {filename}")
            break
    
    # If not found in current directory, try common paths
    if data_path is None:
        common_paths = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Documents"),
        ]
        
        for base_path in common_paths:
            if os.path.exists(base_path):
                for filename in possible_files:
                    full_path = os.path.join(base_path, filename)
                    if os.path.exists(full_path):
                        data_path = full_path
                        print(f"🔍 Found data file: {full_path}")
                        break
                if data_path:
                    break
    
    if data_path is None:
        print("❌ No data file found! Please check these possible names:")
        for name in possible_files[:8]:  # Show first 8 possibilities
            print(f"   • {name}")
        print("\n💡 To use a custom path, create the predictor like this:")
        print("   predictor = HeartDiseasePredictor('your_file_path.csv')")
        print("\n📥 You can download a sample dataset from:")
        print("   https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
        return None
    
    predictor = HeartDiseasePredictor(data_path)
    
    # Run complete analysis
    success = predictor.run_complete_analysis()
    
    if not success:
        print("❌ Analysis failed. Please check your data file.")
        return None
    
    # Example of predicting for new patients
    print("\n" + "="*60)
    print("🔮 EXAMPLE: PREDICTING FOR NEW PATIENTS")
    print("="*60)
    
    # Create example patients based on 5 features
    # Note: These examples use the 5 selected features
    
    # Example patient 1: Higher risk profile
    patient_1 = create_sample_patient_data(predictor, high_risk=True)
    if patient_1:
        print("\n👨 PATIENT 1 (Higher Risk Profile):")
        predictions_1 = predictor.predict_new_patient(patient_1)
    
    # Example patient 2: Lower risk profile  
    patient_2 = create_sample_patient_data(predictor, high_risk=False)
    if patient_2:
        print("\n👩 PATIENT 2 (Lower Risk Profile):")
        predictions_2 = predictor.predict_new_patient(patient_2)
    
    # Show feature importance if Random Forest is available
    if 'Random Forest' in predictor.models:
        print("\n" + "="*60)
        print("🌳 FEATURE IMPORTANCE (Random Forest)")
        print("="*60)
        
        rf_model = predictor.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': predictor.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("📊 Feature Importance (All 5 Features):")
        for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        try:
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], color='lightblue')
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance (Random Forest) - 5 Selected Features', fontweight='bold', fontsize=14)
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ TUTORIAL COMPLETED SUCCESSFULLY!")
    print(f"🎯 Analysis performed using 5 features: {predictor.feature_names}")
    print("="*60)
    
    return predictor

def create_sample_patient_data(predictor, high_risk=True):
    """
    Create sample patient data based on available 5 features
    
    Args:
        predictor: HeartDiseasePredictor instance
        high_risk (bool): Whether to create high-risk or low-risk profile
    
    Returns:
        dict: Sample patient data
    """
    if predictor.feature_names is None:
        return None
    
    patient = {}
    
    for feature in predictor.feature_names:
        feature_lower = feature.lower()
        
        if 'age' in feature_lower:
            patient[feature] = 65 if high_risk else 35
        elif 'chol' in feature_lower:
            patient[feature] = 250 if high_risk else 180
        elif 'chest' in feature_lower and 'pain' in feature_lower:
            patient[feature] = 0 if high_risk else 1  # 0 = Typical Angina (higher risk)
        elif 'angina' in feature_lower:
            patient[feature] = 1 if high_risk else 0  # 1 = Yes (higher risk)
        elif 'slope' in feature_lower or 'st_slope' in feature_lower:
            patient[feature] = 2 if high_risk else 0  # 2 = Downsloping (higher risk)
        else:
            # Default values for unknown features
            patient[feature] = 1 if high_risk else 0
    
    return patient

# ========================================
# HELPER FUNCTIONS
# ========================================

def create_sample_patient():
    """
    Interactive function to create a sample patient for prediction using 5 features
    """
    print("\n🆕 CREATE NEW PATIENT FOR PREDICTION (5 FEATURES)")
    print("-" * 50)
    
    patient = {}
    
    # Questions for the 5 selected features
    questions = {
        'Age': "Enter patient age (20-80): ",
        'Cholesterol': "Enter cholesterol level (100-400, 0 if unknown): ",
        'ChestPainType': "Enter chest pain type (0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal, 3=Asymptomatic): ",
        'ExerciseAngina': "Exercise-induced angina? (0=No, 1=Yes): ",
        'ST_Slope': "Enter ST slope (0=Upsloping, 1=Flat, 2=Downsloping): "
    }
    
    for key, question in questions.items():
        while True:
            try:
                value = input(question).strip()
                
                # Validate input
                if key in ['Age', 'Cholesterol']:
                    patient[key] = int(value)
                elif key in ['ChestPainType', 'ExerciseAngina', 'ST_Slope']:
                    patient[key] = int(value)
                else:
                    patient[key] = value
                
                break
            except ValueError:
                print("❌ Invalid input. Please try again.")
            except KeyboardInterrupt:
                print("\n❌ Input cancelled by user.")
                return None
    
    return patient

def explain_features():
    """
    Explain what each of the 5 selected features means
    """
    print("\n📚 SELECTED 5 FEATURES EXPLANATION")
    print("=" * 45)
    
    explanations = {
        'Age': 'Patient age in years - Higher age increases heart disease risk',
        'Cholesterol': 'Serum cholesterol in mg/dl - Higher levels increase risk (0 if unknown)',
        'ChestPainType': 'Type of chest pain:\n    0: Typical Angina (most concerning)\n    1: Atypical Angina\n    2: Non-Anginal Pain\n    3: Asymptomatic (no chest pain)',
        'ExerciseAngina': 'Exercise-induced angina:\n    1: Yes (higher risk)\n    0: No',
        'ST_Slope': 'Slope of peak exercise ST segment:\n    0: Upsloping (normal)\n    1: Flat (concerning)\n    2: Downsloping (most concerning)'
    }
    
    for i, (feature, explanation) in enumerate(explanations.items(), 1):
        print(f"{i}. {feature:15} : {explanation}")
    
    print("\n💡 Why these 5 features were selected:")
    print("  • Age: Fundamental risk factor for cardiovascular disease")
    print("  • Cholesterol: Key biomarker for heart disease risk")
    print("  • Chest Pain Type: Direct symptom indicator") 
    print("  • Exercise Angina: Important functional assessment")
    print("  • ST Slope: Critical ECG finding during stress testing")

# ========================================
# RUN THE PROGRAM
# ========================================

if __name__ == "__main__":
    """
    Main execution block
    """
    print("🏥 HEART DISEASE PREDICTION SYSTEM (5 FEATURES)")
    print("=" * 55)
    print("This system will help you:")
    print("  1. Analyze heart disease dataset using 5 key features")
    print("  2. Train multiple ML models")
    print("  3. Compare model performance")
    print("  4. Predict for new patients")
    print("\n🎯 Selected Features:")
    print("  • Age")
    print("  • Cholesterol")
    print("  • Chest Pain Type")
    print("  • Exercise Angina")
    print("  • ST Slope")
    print("=" * 55)
    
    try:
        # Run the main tutorial
        predictor = main()
        
        if predictor is not None:
            # Show feature explanations
            explain_features()
            
            print(f"\n🎊 SYSTEM READY FOR USE!")
            print("You can now use the predictor object to make predictions for new patients.")
            print(f"🎯 Model trained on 5 features: {predictor.feature_names}")
            print("\n💡 Example usage:")
            print("  # Create sample patient data")
            print("  patient = {")
            print("      'Age': 60,")
            print("      'Cholesterol': 250,")
            print("      'ChestPainType': 0,")
            print("      'ExerciseAngina': 1,")
            print("      'ST_Slope': 2")
            print("  }")
            print("  # Make prediction")
            print("  predictions = predictor.predict_new_patient(patient)")
            
        else:
            print("\n❌ System initialization failed.")
            print("Please check that you have a valid dataset file.")
            
    except KeyboardInterrupt:
        print("\n\n❌ Program interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your dataset and try again.")