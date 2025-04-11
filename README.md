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
   
# Feedback

After seeing their mentor recommendations, the user can tell us if the suggestions were helpful or not (like a thumbs up/down).

We can collect that feedback and slowly learn which mentors work well with which kinds of students.

If some mentors are always rated low for certain profiles, we can stop showing them to similar users.

Later, we can even use this feedback to train the system to make smarter suggestions without us having to manually adjust anything.

So basically, the more students use it and share what worked for them, the more accurate and useful this mentor recommender will become!


##  How to Run

## Requirements
Install necessary libraries:
```bash

pip install pandas numpy scikit-learn

