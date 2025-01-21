import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.multivariate.manova import MANOVA
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

class EmployeePerformanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Employee Performance Analysis")
        self.geometry("800x600")

        # Create tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.tabs)
        self.tab2 = ttk.Frame(self.tabs)
        self.tab3 = ttk.Frame(self.tabs)

        self.tabs.add(self.tab1, text="Data Exploration")
        self.tabs.add(self.tab2, text="Model Evaluation")
        self.tabs.add(self.tab3, text="Visualizations")

        # Create a text box to display messages
        self.textbox = tk.Text(self, height=3, width=50)
        self.textbox.pack(pady=10)

        # Create a button to load the dataset
        load_data_button = ttk.Button(self, text="Load Dataset", command=self.load_data)
        load_data_button.pack(pady=10)

        # Quit button
        quit_button = ttk.Button(self, text="Quit", command=self.quit_app)
        quit_button.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.textbox.insert(tk.END, "Data Loaded Successfully!\n")

                # Select numeric columns
                self.numeric_columns = ['ATTENDENCE SCORE', 'TENURE (IN YRS)', 'SALARY (INR)', 'ENGAGEMENT SURVEY SCORE', 'TRAINING HOURS',
                               'TEAM PERFORMANCE SCORE', 'COMPANY PERFORMANCE SCORE', 'PERFORMANCE RATINGS']

                self.data_numeric = self.data[self.numeric_columns].copy()

                # Drop rows with missing values
                self.data_numeric.dropna(inplace=True)

                # Define features and target variable
                self.X = self.data_numeric.drop(columns=['PERFORMANCE RATINGS'])  # Features
                self.y = self.data_numeric['PERFORMANCE RATINGS']  # Target variable

                # Split the data into training and testing sets (80% train, 20% test)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

                # Add a constant term to the features for OLS regression
                self.X_train_ols = sm.add_constant(self.X_train)
                self.X_test_ols = sm.add_constant(self.X_test)

                # Fit the OLS model
                self.ols_model = sm.OLS(self.y_train, self.X_train_ols)
                self.ols_results = self.ols_model.fit()

                # Predict using the OLS model
                self.y_pred_ols = self.ols_results.predict(self.X_test_ols)

                # Calculate metrics for OLS model
                self.mse_ols = mean_squared_error(self.y_test, self.y_pred_ols)
                self.rmse_ols = np.sqrt(self.mse_ols)
                self.r2_ols = r2_score(self.y_test, self.y_pred_ols)

                # Multiple Regression model
                self.y_pred_multiple, self.mse_multiple, self.rmse_multiple, self.r2_multiple, self.model_multiple = self.multiple_linear_regression(self.X_train, self.X_test, self.y_train, self.y_test)

                # Perform MANOVA
                self.manova = MANOVA(self.X_train, self.y_train)

                self.create_widgets()
                

            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {e}")

    def multiple_linear_regression(self, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return y_pred, mse, rmse, r2, model

    def create_widgets(self):
        # Tab 1: Data Exploration
        self.create_data_exploration_tab()

        # Tab 2: Model Evaluation
        self.create_model_evaluation_tab()

        # Tab 3: Visualizations
        self.create_visualizations_tab()

    def create_data_exploration_tab(self):
        # Add widgets and layouts for data exploration
        data_exploration_frame = ttk.Frame(self.tab1)
        data_exploration_frame.pack(side="left",fill="both", expand=True, padx=20, pady=20)

        # Display the dataset
        data_tree = ttk.Treeview(data_exploration_frame)
        data_tree["columns"] = list(self.data.columns)
        data_tree.column("#0", width=100)
        for col in data_tree["columns"]:
            data_tree.column(col, width=100)
        data_tree.pack(fill="both", expand=True)

        # Populate the data tree
        for i, row in self.data.iterrows():
            data_tree.insert("", "end", text=str(i), values=list(row))

        '''  # Adjust the width and height of the Treeview widget
        data_tree.config(height = 10, width = 30)  # Adjust the values as needed '''

    def create_model_evaluation_tab(self):
        # Add widgets and layouts for model evaluation
        model_evaluation_frame = ttk.Frame(self.tab2)
        model_evaluation_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Display the OLS regression results
        ols_results_text = tk.Text(model_evaluation_frame, height=10, width=80)
        ols_results_text.pack(fill="both", expand=True)
        ols_results_text.insert("1.0", str(self.ols_results.summary()))

        # Display the OLS model metrics
        ols_metrics_label = ttk.Label(model_evaluation_frame, text="OLS Model Metrics:")
        ols_metrics_label.pack(anchor="w", pady=10)

        ols_mse_label = ttk.Label(model_evaluation_frame, text=f"Mean Squared Error: {self.mse_ols:.2f}")
        ols_mse_label.pack(anchor="w")

        ols_rmse_label = ttk.Label(model_evaluation_frame, text=f"Root Mean Squared Error: {self.rmse_ols:.2f}")
        ols_rmse_label.pack(anchor="w")

        ols_r2_label = ttk.Label(model_evaluation_frame, text=f"R-squared: {self.r2_ols:.2f}")
        ols_r2_label.pack(anchor="w")

        # Display the Multiple Regression model metrics
        multiple_metrics_label = ttk.Label(model_evaluation_frame, text="Multiple Regression Model Metrics:")
        multiple_metrics_label.pack(anchor="w", pady=10)

        multiple_mse_label = ttk.Label(model_evaluation_frame, text=f"Mean Squared Error: {self.mse_multiple:.2f}")
        multiple_mse_label.pack(anchor="w")

        multiple_rmse_label = ttk.Label(model_evaluation_frame, text=f"Root Mean Squared Error: {self.rmse_multiple:.2f}")
        multiple_rmse_label.pack(anchor="w")

        multiple_r2_label = ttk.Label(model_evaluation_frame, text=f"R-squared: {self.r2_multiple:.2f}")
        multiple_r2_label.pack(anchor="w")

        # Display the MANOVA results
        manova_results_text = tk.Text(model_evaluation_frame, height=5, width=80)
        manova_results_text.pack(fill="both", expand=True, pady=20)
        manova_results_text.insert("1.0", str(self.manova.mv_test()))


    def create_visualizations_tab(self):
        # Add widgets and layouts for visualizations
        visualizations_frame = ttk.Frame(self.tab3)
        visualizations_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create tabs for each plot
        tabs = ttk.Notebook(visualizations_frame)
        tabs.pack(fill="both", expand=True)

        # Add tabs for each plot
        self.create_correlation_matrix_tab(tabs)
        self.create_actual_vs_predicted_tab(tabs)
        self.create_residuals_plot_tab(tabs)
        self.create_qq_plot_tab(tabs)

    def create_correlation_matrix_tab(self, tabs):
        # Create a tab for the correlation matrix plot
        correlation_tab = ttk.Frame(tabs)
        tabs.add(correlation_tab, text="Correlation Matrix")

        # Correlation Matrix plot
        correlation_matrix_figure = plt.figure(figsize=(6, 4))
        sns.heatmap(self.data_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=correlation_matrix_figure.add_subplot(111))
        correlation_matrix_figure.suptitle('Correlation Matrix')

        correlation_matrix_canvas = FigureCanvasTkAgg(correlation_matrix_figure, master=correlation_tab)
        correlation_matrix_canvas.draw()
        correlation_matrix_canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_actual_vs_predicted_tab(self, tabs):
        # Create a tab for the Actual vs Predicted plot
        actual_vs_predicted_tab = ttk.Frame(tabs)
        tabs.add(actual_vs_predicted_tab, text="Actual vs Predicted")

        # Actual vs Predicted Performance Ratings plot
        actual_vs_predicted_figure = plt.figure(figsize=(6, 4))
        plt.scatter(self.y_test, self.y_pred_multiple, color='blue')
        plt.xlabel('Actual Performance Ratings')
        plt.ylabel('Predicted Performance Ratings (Multiple Regression)')
        plt.title('Actual vs Predicted Performance Ratings (Multiple Regression)')
        plt.grid(True)

        actual_vs_predicted_canvas = FigureCanvasTkAgg(actual_vs_predicted_figure, master=actual_vs_predicted_tab)
        actual_vs_predicted_canvas.draw()
        actual_vs_predicted_canvas.get_tk_widget().pack(fill="both", expand=True)


    def create_residuals_plot_tab(self, tabs):
        # Create a tab for the Residuals plot
       residuals_tab = ttk.Frame(tabs)
       tabs.add(residuals_tab, text="Residuals")
    
       # Residuals plot
       residuals_figure = plt.figure(figsize=(10, 6))
       plt.scatter(self.y_pred_ols, self.y_test - self.y_pred_ols, color='blue')
       plt.xlabel('Predicted Performance Ratings (OLS)')
       plt.ylabel('Residuals (OLS)')
       plt.title('Residual Analysis (OLS)')
       plt.axhline(y=0, color='r', linestyle='-')
       plt.grid(True)

       residuals_canvas = FigureCanvasTkAgg(residuals_figure, master=residuals_tab)
       residuals_canvas.draw()
       residuals_canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_qq_plot_tab(self, tabs):
        # Create a tab for the Q-Q plot
        qq_tab = ttk.Frame(tabs)
        tabs.add(qq_tab, text="Q-Q Plot")

        # Q-Q plot
        qq_figure = plt.figure(figsize=(10, 6))
        ax = qq_figure.add_subplot(111)
        qqplot(self.y_test - self.y_pred_ols, line='s', ax=ax)
        ax.set_title('Normal Q-Q Plot (OLS)')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.grid(True)

        qq_canvas = FigureCanvasTkAgg(qq_figure, master=qq_tab)
        qq_canvas.draw()
        qq_canvas.get_tk_widget().pack(fill="both", expand=True)


    def quit_app(self):
        self.destroy()
        
    

if __name__ == "__main__":
    app = EmployeePerformanceApp()
    app.mainloop()
