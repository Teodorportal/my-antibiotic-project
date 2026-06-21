# antibiotic_resistance_detection.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


class AntibioticResistanceDetector:
    """
    This class loads or creates antibiotic resistance data,
    preprocesses it, trains a machine learning model,
    evaluates the model, and plots useful charts.
    """

    def __init__(self, file_path="antibiotic_data.csv"):
        self.file_path = file_path
        self.data = None
        self.model = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            max_depth=8
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = "Resistance"

    def create_sample_dataset(self, n_samples=300):
        """
        Creates a sample dataset if no external CSV file exists.
        This makes the project executable without needing a real dataset.
        """

        np.random.seed(42)

        bacteria_list = ["E.coli", "Klebsiella", "Staphylococcus", "Pseudomonas"]
        antibiotic_list = ["Ciprofloxacin", "Amoxicillin", "Gentamicin", "Ceftriaxone"]
        hospital_wards = ["ICU", "Emergency", "Surgery", "Internal"]

        rows = []

        for _ in range(n_samples):
            bacteria = np.random.choice(bacteria_list)
            antibiotic = np.random.choice(antibiotic_list)
            ward = np.random.choice(hospital_wards)

            age = np.random.randint(1, 90)
            previous_use = np.random.choice([0, 1], p=[0.55, 0.45])

            mic = round(np.random.uniform(0.1, 64.0), 2)
            inhibition_zone = round(np.random.uniform(5.0, 35.0), 2)

            resistance_score = 0

            if mic > 16:
                resistance_score += 2
            elif mic > 8:
                resistance_score += 1

            if inhibition_zone < 15:
                resistance_score += 2
            elif inhibition_zone < 20:
                resistance_score += 1

            if previous_use == 1:
                resistance_score += 1

            if ward == "ICU":
                resistance_score += 1

            if bacteria in ["Pseudomonas", "Klebsiella"]:
                resistance_score += 1

            if antibiotic in ["Amoxicillin", "Ceftriaxone"]:
                resistance_score += 1

            if resistance_score >= 4:
                resistance = "Resistant"
            else:
                resistance = "Sensitive"

            rows.append([
                bacteria,
                antibiotic,
                ward,
                age,
                previous_use,
                mic,
                inhibition_zone,
                resistance
            ])

        self.data = pd.DataFrame(rows, columns=[
            "Bacteria",
            "Antibiotic",
            "Ward",
            "Age",
            "Previous_Antibiotic_Use",
            "MIC",
            "Inhibition_Zone",
            "Resistance"
        ])

        self.data.to_csv(self.file_path, index=False)
        print(f"Sample dataset created and saved as {self.file_path}")

    def load_data(self):
        """
        Loads data from CSV file.
        If the file does not exist, it creates a sample dataset.
        """

        if not os.path.exists(self.file_path):
            print("Dataset not found. Creating sample dataset...")
            self.create_sample_dataset()

        self.data = pd.read_csv(self.file_path)
        print("Dataset loaded successfully.")
        print("\nFirst 5 rows:")
        print(self.data.head())

    def show_basic_info(self):
        """
        Displays basic information about the dataset.
        """

        print("\nDataset Information:")
        print(self.data.info())

        print("\nStatistical Description:")
        print(self.data.describe())

        print("\nResistance Count:")
        print(self.data[self.target_column].value_counts())

    def preprocess_data(self):
        """
        Encodes categorical columns and scales numerical features.
        """

        df = self.data.copy()

        categorical_columns = ["Bacteria", "Antibiotic", "Ward", "Resistance"]

        for col in categorical_columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.label_encoders[col] = encoder

        self.feature_columns = [
            "Bacteria",
            "Antibiotic",
            "Ward",
            "Age",
            "Previous_Antibiotic_Use",
            "MIC",
            "Inhibition_Zone"
        ]

        X = df[self.feature_columns]
        y = df[self.target_column]

        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(
            X_scaled,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

    def train_model(self, X_train, y_train):
        """
        Trains the Random Forest classifier.
        """

        self.model.fit(X_train, y_train)
        print("\nModel training completed.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained model and prints accuracy and report.
        """

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print("\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)

        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """
        Plots confusion matrix.
        """

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = self.label_encoders[self.target_column].classes_
        tick_marks = np.arange(len(classes))

        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

    def plot_resistance_distribution(self):
        """
        Plots the distribution of resistant and sensitive samples.
        """

        counts = self.data[self.target_column].value_counts()

        plt.figure(figsize=(6, 5))
        counts.plot(kind="bar", color=["tomato", "seagreen"])
        plt.title("Distribution of Antibiotic Resistance")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig("resistance_distribution.png")
        plt.show()

    def plot_antibiotic_resistance(self):
        """
        Plots resistance count based on antibiotic type.
        """

        grouped = self.data.groupby(["Antibiotic", "Resistance"]).size().unstack()

        grouped.plot(kind="bar", figsize=(9, 5))
        plt.title("Resistance Status by Antibiotic")
        plt.xlabel("Antibiotic")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("antibiotic_resistance_chart.png")
        plt.show()

    def plot_mic_vs_zone(self):
        """
        Plots relationship between MIC and inhibition zone.
        """

        colors = {
            "Resistant": "red",
            "Sensitive": "green"
        }

        plt.figure(figsize=(8, 5))

        for label in self.data[self.target_column].unique():
            subset = self.data[self.data[self.target_column] == label]
            plt.scatter(
                subset["MIC"],
                subset["Inhibition_Zone"],
                label=label,
                alpha=0.7,
                color=colors.get(label, "blue")
            )

        plt.title("MIC vs Inhibition Zone")
        plt.xlabel("MIC")
        plt.ylabel("Inhibition Zone Diameter")
        plt.legend()
        plt.tight_layout()
        plt.savefig("mic_vs_zone.png")
        plt.show()

    def predict_new_sample(self):
        """
        Gets a sample input from the user and predicts resistance status.
        """

        print("\nNew Sample Prediction")
        print("Available Bacteria: E.coli, Klebsiella, Staphylococcus, Pseudomonas")
        print("Available Antibiotics: Ciprofloxacin, Amoxicillin, Gentamicin, Ceftriaxone")
        print("Available Wards: ICU, Emergency, Surgery, Internal")

        bacteria = input("Enter bacteria name: ")
        antibiotic = input("Enter antibiotic name: ")
        ward = input("Enter hospital ward: ")
        age = int(input("Enter patient age: "))
        previous_use = int(input("Previous antibiotic use? 0 = No, 1 = Yes: "))
        mic = float(input("Enter MIC value: "))
        inhibition_zone = float(input("Enter inhibition zone diameter: "))

        sample = pd.DataFrame([[
            bacteria,
            antibiotic,
            ward,
            age,
            previous_use,
            mic,
            inhibition_zone
        ]], columns=self.feature_columns)

        for col in ["Bacteria", "Antibiotic", "Ward"]:
            sample[col] = self.label_encoders[col].transform(sample[col])

        sample_scaled = self.scaler.transform(sample)
        prediction = self.model.predict(sample_scaled)[0]

        result = self.label_encoders[self.target_column].inverse_transform([prediction])[0]

        print("\nPrediction Result:")
        print(f"The sample is predicted as: {result}")


def main():
    detector = AntibioticResistanceDetector()

    detector.load_data()
    detector.show_basic_info()

    X_train, X_test, y_train, y_test = detector.preprocess_data()

    detector.train_model(X_train, y_train)
    detector.evaluate_model(X_test, y_test)

    detector.plot_resistance_distribution()
    detector.plot_antibiotic_resistance()
    detector.plot_mic_vs_zone()

    answer = input("\nDo you want to predict a new sample? y/n: ")

    if answer.lower() == "y":
        detector.predict_new_sample()

    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()
