# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# class DataCleaner:
#     def __init__(self, df):
#         self.df = df.copy()
#         self.logs = []
#         self.encoded_columns = []
#         self.onehot_encoder = None
#         self.ordinal_encoder = None

#     def log(self, message):
#         print(message)
#         self.logs.append(message)

#     def is_id_or_name(self, col):
#         return any(keyword in col.lower() for keyword in ['id', 'name'])

#     def preview(self):
#         self.log("\nPreview of dataset:")
#         self.log(str(self.df.head()))

#     def describe(self):
#         self.log("\nDataset description:")
#         self.log(str(self.df.describe(include='all')))

#     def handle_missing_values(self):
#         self.log("\nHandling missing values:")
#         for col in self.df.columns:
#             if self.df[col].isnull().sum() > 0:
#                 if self.df[col].dtype in [np.float64, np.int64]:
#                     self.df[col].fillna(self.df[col].median(), inplace=True)
#                     self.log(f"Filled missing numeric values in '{col}' with median.")
#                 else:
#                     self.df[col].fillna(self.df[col].mode()[0], inplace=True)
#                     self.log(f"Filled missing categorical values in '{col}' with mode.")

#     def remove_duplicates(self):
#         initial = len(self.df)
#         self.df.drop_duplicates(inplace=True)
#         removed = initial - len(self.df)
#         self.log(f"\nRemoved {removed} duplicate rows.")

#     def visualize_histograms(self):
#         self.log("\nSaving histograms for numerical features (excluding ID/name-like columns)...")
#         num_cols = self.df.select_dtypes(include=[np.number]).columns
#         plot_cols = [col for col in num_cols if not self.is_id_or_name(col)]

#         if not plot_cols:
#             self.log("No numeric columns to plot histograms.")
#             return

#         self.df[plot_cols].hist(bins=50, figsize=(20, 15))
#         plt.tight_layout()
#         plt.savefig("all_numeric_histograms.png")
#         plt.close()
#         self.log("Saved all histograms to 'all_numeric_histograms.png'.")

#     def visualize_correlation(self):
#         self.log("\nSaving correlation heatmap...")
#         num_cols = self.df.select_dtypes(include=[np.number]).columns
#         if len(num_cols) < 2:
#             self.log("Not enough numeric columns for correlation heatmap.")
#             return

#         corr_matrix = self.df[num_cols].corr()
#         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
#         plt.title("Correlation Heatmap")
#         plt.tight_layout()
#         plt.savefig("correlation_heatmap.png")
#         plt.close()
#         self.log("Correlation heatmap saved to 'correlation_heatmap.png'.")

#     def encode_features(self, mode="auto"):
#         self.log("\nEncoding categorical features (safe rules):")
#         cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
#         cat_cols = [col for col in cat_cols if not self.is_id_or_name(col)]

#         if len(cat_cols) == 0:
#             self.log("No categorical features to encode.")
#             return

#         # Ordinal encoding (for demonstration)
#         self.ordinal_encoder = OrdinalEncoder()
#         ord_encoded = self.ordinal_encoder.fit_transform(self.df[cat_cols])
#         self.log(f"Ordinal encoded {len(cat_cols)} columns. Categories: {self.ordinal_encoder.categories_}")

#         # One-hot encoding
#         self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         onehot_encoded = self.onehot_encoder.fit_transform(self.df[cat_cols])
#         encoded_df = pd.DataFrame(onehot_encoded, columns=self.onehot_encoder.get_feature_names_out(cat_cols), index=self.df.index)

#         self.df.drop(columns=cat_cols, inplace=True)
#         self.df = pd.concat([self.df, encoded_df], axis=1)
#         self.encoded_columns.extend(encoded_df.columns)

#         self.log(f"One-hot encoded {len(cat_cols)} categorical columns. Added {encoded_df.shape[1]} new features.")

#     def save_cleaned_data(self, path="cleaned_data.csv", format="csv"):
#         self.log(f"\nSaving cleaned dataset as {path}...")
#         if format == "csv":
#             self.df.to_csv(path, index=False)
#         elif format == "json":
#             self.df.to_json(path, orient="records", lines=True)
#         elif format == "excel":
#             self.df.to_excel(path, index=False)
#         else:
#             self.log("Unsupported format. Use csv, json, or excel.")
#             return
#         self.log(f"Dataset saved to {path}.")

#     def save_encoders(self, onehot_path="onehot_encoder.pkl", ordinal_path="ordinal_encoder.pkl"):
#         if self.onehot_encoder:
#             joblib.dump(self.onehot_encoder, onehot_path)
#             self.log(f"OneHotEncoder saved to {onehot_path}.")
#         if self.ordinal_encoder:
#             joblib.dump(self.ordinal_encoder, ordinal_path)
#             self.log(f"OrdinalEncoder saved to {ordinal_path}.")

#     def save_report(self, path="cleaning_report.txt"):
#         self.log(f"\nSaving cleaning report to {path}...")
#         with open(path, "w") as f:
#             for log in self.logs:
#                 f.write(log + "\n")
#         self.log("Report saved.")

#     def run_pipeline(self):
#         self.preview()
#         self.describe()
#         self.visualize_histograms()
#         self.visualize_correlation()
#         self.handle_missing_values()
#         self.remove_duplicates()
#         self.encode_features()
#         self.save_cleaned_data("cleaned_data.csv")
#         self.save_encoders()
#         self.save_report("cleaning_report.txt")
#         self.log("\nPipeline completed.")
    
    # Return both the cleaned DataFrame and the log report
        # return self.df, self.logs  
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import numpy as np

# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# class DataCleaner:
    

#     def __init__(self, df: pd.DataFrame, verbose=True):
#         self.original_df = df
#         self.df = df.copy()
#         self.verbose = verbose
#         self.logs = []
#         self.histograms = {}
#         self.correlation_matrix = None
#     def is_id_or_name(self, col):
#         col_lower = col.lower()
#         return any(keyword in col_lower for keyword in ['id', 'name', 'identifier'])

#     def run_pipeline(self):
#         self._drop_duplicates()
#         self._handle_missing_values()
#         self.encode_features()
#         self._generate_histograms()
#         self._compute_correlation()
#         self._log.()
#         return self.df, self.histograms, self.heatmap, self.report

#     def _log(self, message):
#         if self.verbose:
#             print(message)
#         self.logs.append(message)
  

#     def _drop_duplicates(self):
#         before = len(self.df)
#         self.df.drop_duplicates(inplace=True)
#         after = len(self.df)
#         self._log(f"Dropped {before - after} duplicate rows.")

#     def _handle_missing_values(self):
#         missing = self.df.isnull().sum().sum()
#         self.df.fillna(self.df.mode().iloc[0], inplace=True)
#         self._log(f"Filled {missing} missing values using mode.")

#     def encode_features(self):
#         self.encoded_columns = []
#         self._log(" Encoding categorical features (ordinal for <8 categories)...")
#         cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
#         cat_cols = [col for col in cat_cols if not self.is_id_or_name(col)]

#         if not cat_cols:
#             self._log("No categorical features to encode.")
#             return

#         ordinal_cols = [col for col in cat_cols if self.df[col].nunique() <= 8]
#         onehot_cols = [col for col in cat_cols if col not in ordinal_cols]

#         if ordinal_cols:
#             self.ordinal_encoder = OrdinalEncoder()
#             self.df[ordinal_cols] = self.ordinal_encoder.fit_transform(self.df[ordinal_cols])
#             self.encoded_columns.extend(ordinal_cols)
#             self._log(f"Ordinal encoded: {ordinal_cols}")

#         if onehot_cols:
#             self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#             onehot_encoded = self.onehot_encoder.fit_transform(self.df[onehot_cols])
#             encoded_df = pd.DataFrame(onehot_encoded,
#                                       columns=self.onehot_encoder.get_feature_names_out(onehot_cols),
#                                       index=self.df.index)
#             self.df.drop(columns=onehot_cols, inplace=True)
#             self.df = pd.concat([self.df, encoded_df], axis=1)
#             self.encoded_columns.extend(encoded_df.columns.tolist())
#             self._log(f"One-hot encoded: {onehot_cols} into {list(encoded_df.columns)}")


#     def _generate_histograms(self):
#         numeric_cols = self.df.select_dtypes(include='number').columns
#         for col in numeric_cols:
#             fig, ax = plt.subplots()
#             self.df[col].hist(ax=ax, bins=20)
#             ax.set_title(f"Histogram of {col}")
#             ax.set_xlabel(col)
#             ax.set_ylabel("Frequency")
#             buf = io.BytesIO()
#             plt.savefig(buf, format="png")
#             buf.seek(0)
#             self.histograms[col] = buf
#             plt.close(fig)
#     def _compute_correlation(self):
#         numeric_df = self.df.select_dtypes(include='number')
#         if numeric_df.shape[1] < 2:
#             self._log("Not enough numeric columns to compute correlation.")
#             return
#         corr_matrix = numeric_df.corr().round(2)
#         corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))  # Remove self-correlation
#         self.correlation_matrix = corr_matrix

#         # Save heatmap
#         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#         ax.set_title("Correlation Heatmap")
#         buf = io.BytesIO()
#         plt.tight_layout()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         self.heatmap = buf
#         plt.close(fig)
#         self._log("Correlation heatmap generated.")
#     # def _compute_correlation(self):
#     #     numeric_df = self.df.select_dtypes(include='number')
#     #     corr_matrix = numeric_df.corr().round(2)
#     #     corr_matrix = corr_matrix.where(~(corr_matrix == 1.0))  # Remove self-correlation
#     #     self.correlation_matrix = corr_matrix
# ***************************************************************************************************8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class DataCleaner:

    def __init__(self, df: pd.DataFrame, verbose=True):
        self.original_df = df
        self.df = df.copy()
        self.verbose = verbose
        self.logs = []
        self.histograms = {}
        self.correlation_matrix = None
        self.heatmap = None
        self.report = ""

    def is_id_or_name(self, col):
        col_lower = col.lower()
        return any(keyword in col_lower for keyword in ['id', 'name', 'identifier'])

    def run_pipeline(self):
        self._drop_duplicates()
        self._handle_missing_values()
        self.encode_features()
        self._generate_histograms()
        self._compute_correlation()
        self._save_report()
        return self.df, self.histograms, self.heatmap, self.report

    def _log(self, message):
        if self.verbose:
            print(message)
        self.logs.append(message)

    def _save_report(self):
        self.report = "\n".join(self.logs)
        self._log("Cleaning report ready to view and download.")

    def _drop_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        self._log(f"Dropped {before - after} duplicate rows.")

    def _handle_missing_values(self):
        missing = self.df.isnull().sum().sum()
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        self._log(f"Filled {missing} missing values using mode.")

    def encode_features(self):
        self.encoded_columns = []
        self._log("Encoding categorical features (ordinal for <8 categories)...")
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if not self.is_id_or_name(col)]

        if not cat_cols:
            self._log("No categorical features to encode.")
            return

        ordinal_cols = [col for col in cat_cols if self.df[col].nunique() <= 8]
        onehot_cols = [col for col in cat_cols if col not in ordinal_cols]

        if ordinal_cols:
            self.ordinal_encoder = OrdinalEncoder()
            self.df[ordinal_cols] = self.ordinal_encoder.fit_transform(self.df[ordinal_cols])
            self.encoded_columns.extend(ordinal_cols)
            self._log(f"Ordinal encoded: {ordinal_cols}")

        if onehot_cols:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_encoded = self.onehot_encoder.fit_transform(self.df[onehot_cols])
            encoded_df = pd.DataFrame(onehot_encoded,
                                      columns=self.onehot_encoder.get_feature_names_out(onehot_cols),
                                      index=self.df.index)
            self.df.drop(columns=onehot_cols, inplace=True)
            self.df = pd.concat([self.df, encoded_df], axis=1)
            self.encoded_columns.extend(encoded_df.columns.tolist())
            self._log(f"One-hot encoded: {onehot_cols} into {list(encoded_df.columns)}")

    def _generate_histograms(self):
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            self.df[col].hist(ax=ax, bins=20)
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            self.histograms[col] = buf
            plt.close(fig)
            self._log(f"Histogram generated for {col}")
            


    def _compute_correlation(self):
        numeric_df = self.df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            self._log("Not enough numeric columns to compute correlation.")
            return
        corr_matrix = numeric_df.corr().round(2)
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))  # Remove self-correlation
        self.correlation_matrix = corr_matrix

        # Save heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        self.heatmap = buf
        plt.close(fig)
        self._log("Correlation heatmap generated.")
