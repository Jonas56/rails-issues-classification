import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

github_api_url = 'https://api.github.com/repos/rails/rails/issues'
per_page = 100
total_issues = 500
model_name = './github_issue_classifier'

def fetch_issues(page):
    params = {
        'state': 'all',
        'per_page': per_page,
        'page': page
    }
    response = requests.get(github_api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'Failed to fetch data: {response.status_code} - {response.text}')

def fetch_all_issues():
    all_issues = []
    for page in range(1, (total_issues // per_page) + 2):
        issues = fetch_issues(page)
        all_issues.extend(issues)
        if len(issues) < per_page or len(all_issues) >= total_issues:
            break
    return all_issues[:total_issues]

def fetch_issues_labels():
    data = fetch_all_issues()
    issues = []
    for issue in data:
        if not issue['labels']:
            continue
        issues.append({
            'labels': [label['name'] for label in issue['labels']],
        })
    return pd.DataFrame(issues)

def predict_label(text):
    mlb = MultiLabelBinarizer()
    issues_df = fetch_issues_labels()
    mlb.fit_transform(issues_df['labels'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    return mlb.classes_[predicted_class_id]

text = input("Enter the issue title & description: ")
prediction = predict_label(text)
print(f"Predicted label: {prediction}")
