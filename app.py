import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

np.random.seed(42)
X = np.random.rand(200, 2) * [50, 100]  # Age (0-50), Salary (0-100k)
y = (X[:, 0] + X[:, 1] > 100).astype(int)  # Purchase decision: 1 if Age + Salary > 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")


def predict_purchase(age, salary):
    try:
        age = float(age)
        salary = float(salary)
        prediction = model.predict([[age, salary]])[0]
        return "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"
    except ValueError:
        return "Invalid input. Please enter numeric values for both Age and Salary."


interface = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Textbox(label="Age", placeholder="Enter age (e.g., 25)"),
        gr.Textbox(label="Salary", placeholder="Enter salary (e.g., 50000)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Purchase Prediction Model",
    description="Predicts whether a person is likely to purchase a product based on their age and salary.",
)

# Step 5: Launch the app
if __name__ == "__main__":
    interface.launch()
