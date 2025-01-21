import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Linear Regression Model")

        self.label = tk.Label(master, text="Select CSV File:")
        self.label.grid(row=0, column=0)

        self.browse_button = tk.Button(master, text="Browse", command=self.load_data)
        self.browse_button.grid(row=0, column=1)

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.correlation_button = tk.Button(master, text="Correlation Matrix", command=self.plot_correlation)
        self.correlation_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.scatter_button = tk.Button(master, text="Scatter Plot", command=self.plot_scatter)
        self.scatter_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.quit_button = tk.Button(master, text="Quit", command=self.quit_window)
        self.quit_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.textbox = tk.Text(master, height=10, width=50)
        self.textbox.grid(row=6, column=0, columnspan=2, padx=10, pady=10)



        

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.textbox.insert(tk.END, "Data Loaded Successfully!\n")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {e}")

    def train_model(self):
        if hasattr(self, 'data'):
            numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE',
                               'TRAINING HOURS', 'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE',
                               'PERFORMANCE RATINGS']
            data_numeric = self.data[numeric_columns].dropna()
            X = data_numeric.drop(columns=['PERFORMANCE RATINGS'])
            y = data_numeric['PERFORMANCE RATINGS']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            self.textbox.insert(tk.END, "Model Trained Successfully!\n")
        else:
            messagebox.showerror("Error", "No data loaded!")

    def predict(self):
        if hasattr(self, 'model'):
            if hasattr(self, 'data'):
                numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE',
                                   'TRAINING HOURS', 'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE',
                                   'PERFORMANCE RATINGS']
                data_numeric = self.data[numeric_columns].dropna()
                X = data_numeric.drop(columns=['PERFORMANCE RATINGS'])
                y = data_numeric['PERFORMANCE RATINGS']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                self.textbox.insert(tk.END, f"Mean Squared Error: {mse}\n")
                self.textbox.insert(tk.END, f"Root Mean Squared Error: {rmse}\n")
                self.textbox.insert(tk.END, f"R-squared: {r2}\n")
            else:
                messagebox.showerror("Error", "No data loaded!")
        else:
            messagebox.showerror("Error", "Model not trained!")

    def plot_correlation(self):
        if hasattr(self, 'data'):
            numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE',
                               'TRAINING HOURS', 'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE',
                               'PERFORMANCE RATINGS']
            data_numeric = self.data[numeric_columns].dropna()
            plt.figure(figsize=(12, 6))
            sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.show()
        else:
            messagebox.showerror("Error", "No data loaded!")

    def plot_scatter(self):
        if hasattr(self, 'data'):
            numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE',
                               'TRAINING HOURS', 'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE',
                               'PERFORMANCE RATINGS']
            data_numeric = self.data[numeric_columns].dropna()
            X = data_numeric.drop(columns=['PERFORMANCE RATINGS'])
            y = data_numeric['PERFORMANCE RATINGS']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = self.model.predict(X_test)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, color='blue')
            plt.xlabel('Actual Performance Ratings')
            plt.ylabel('Predicted Performance Ratings')
            plt.title('Actual vs Predicted Performance Ratings')
            plt.show()
        else:
            messagebox.showerror("Error", "No data loaded!")

    def quit_window(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = LinearRegressionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
