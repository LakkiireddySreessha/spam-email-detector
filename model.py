import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# STEP 1: Load data
df = pd.read_csv(
    "spam.csv",
    sep='\t',
    header=None,
    names=['label', 'message'],
    encoding='ISO-8859-1'
)

# STEP 2: Add all custom spam data (ALL IN ONE)
custom_spam = pd.DataFrame({
    'label': ['spam'] * 22,
    'message': [
        'your friend sent you money click to receive',
        'click here to claim your reward now',
        'you have received a transfer click to accept',
        'urgent action required claim your money now',
        'someone sent you cash click to claim now',
        'you got money from a friend click link to receive',
        'receive your payment now click here to accept',
        'you have been sent money click to view',
        'claim the money sent to you now',
        'your account has received funds click to check',
        'money is waiting for you click here',
        'click the link to access your transferred money',
        'you received a payment click here to accept',
        'transfer received from your friend click to claim',
        'urgent! verify your payment to receive funds',
        'someone sent you money claim it now',
        'click here to access your received cash',
        'you have been sent money verify now',
        'money has been deposited click to claim',
        'payment received from unknown source click link',
        'funds waiting in your account click to view',
        'you got money from friend click to accept'
    ]
})

df = pd.concat([df, custom_spam], ignore_index=True)

# STEP 3: Clean + convert labels safely
df['label'] = df['label'].astype(str).str.lower().str.strip()
df = df.dropna(subset=['label'])                 # remove empty labels
df = df[df['label'].isin(['ham','spam'])]       # keep only valid labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# STEP 4: Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['message'] = df['message'].apply(clean_text)

# STEP 5: TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)

X = tfidf.fit_transform(df['message'])
y = df['label']

# STEP 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 7: Train model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# STEP 8: Evaluate
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Threshold Testing ===")
for t in [0.2, 0.3, 0.5]:
    y_pred = (y_prob > t).astype(int)
    print(f"\nThreshold: {t}")
    print(classification_report(y_test, y_pred))

# STEP 9: Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")

# STEP 10: Test samples
samples = [
    "are you coming to class today",
    "urgent! you have won 1000 dollars click now",
    "hey call me when you are free",
    "congratulations you won a free iphone click here",
    "your friend sent you money click to receive"
]

samples_clean = [clean_text(s) for s in samples]
samples_tfidf = tfidf.transform(samples_clean)

probs = model.predict_proba(samples_tfidf)[:, 1]
preds = (probs > 0.3).astype(int)

for i in range(len(samples)):
    print("\nMessage:", samples[i])
    print("Probability:", round(probs[i], 3))
    print("Prediction:", "Spam" if preds[i]==1 else "Not Spam")