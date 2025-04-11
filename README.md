#  Mentor Recommendation System using KNN

This project is a simple AI/ML-based solution to help CLAT aspirants find the most suitable mentors based on their preferences. It was built for the NLTI Internship Assignment (Task 1).


##  Objective

Recommend the top 3 mentors to an aspirant based on:
- Preferred subjects
- Target college
- Learning style

The system uses:
- Cosine similarity
- K-Nearest Neighbors (KNN)
- Preprocessed features using encoding techniques

##  Project Files

- `task1_knn_mentor_recommendation_user_input.py`: Python script that runs in terminal. Accepts user input and prints top mentor recommendations.
- `mentors_data` is hardcoded as sample data. You may replace or expand it for testing.


## How It Works

1. User inputs:
   - Preferred subjects (multi-label)
   - Target law college
   - Learning style
2. These features are encoded numerically:
   - Subjects: `MultiLabelBinarizer`
   - College & Style: `One-Hot Encoding`
3. KNN with cosine similarity compares the aspirant's vector to mentor profiles.
4. Outputs top 3 most similar mentors.

## ▶️ How to Run

### Requirements
Install necessary libraries:
```bash
pip install pandas numpy scikit-learn
