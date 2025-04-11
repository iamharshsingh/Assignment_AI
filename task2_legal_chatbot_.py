
# Task 2: Legal Exam Query Chatbot using KNN (Jupyter Notebook Version)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Step 1: Knowledge Base
# ------------------------------
knowledge_base = {
    "syllabus clat 2025": "The CLAT 2025 syllabus includes English, Legal Reasoning, Logical Reasoning, Current Affairs, and Quantitative Techniques.",
    "cut off nlsiu": "The cut-off for NLSIU Bangalore in 2024 was around AIR 80 for general category.",
    "english section questions": "The English section typically has around 28–32 questions.",
    "legal reasoning tips": "Focus on comprehension and logical application of principles for legal reasoning questions.",
    "clat exam duration": "The CLAT exam duration is 2 hours.",
    "marks per question": "Each question carries 1 mark. There is a 0.25 mark negative for wrong answers."
}

# ------------------------------
# Step 2: Display Topics
# ------------------------------
print("📘 CLAT FAQ Topics Available:")
for question in knowledge_base:
    print("-", question)

# ------------------------------
# Step 3: User Query
# ------------------------------
user_query = input("\nAsk your CLAT-related question: ").lower()

# ------------------------------
# Step 4: TF-IDF Vectorization
# ------------------------------
faq_questions = list(knowledge_base.keys())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_questions)
user_vec = vectorizer.transform([user_query])

# ------------------------------
# Step 5: Use KNN to find the closest question
# ------------------------------
model = NearestNeighbors(n_neighbors=1, metric='cosine')
model.fit(X)
distance, index = model.kneighbors(user_vec)

best_question = faq_questions[index[0][0]]
best_answer = knowledge_base[best_question]

# ------------------------------
# Step 6: Show Answer
# ------------------------------
print("\n✅ Closest Match:", best_question)
print("💬 Answer:", best_answer)
