# Submission


In this, I tried to predict whether someone will get their home loan approved or not. The idea is to use machine learning to learn from old data, and then apply the model to new cases. Along with this, I also checked if there's any unfairness in the decisions, mainly based on gender.

I worked with a dataset which contains people’s information like their income, credit score, age, gender, race, job type, education and if they got loan approved or not. First, I loaded the data from a CSV file into Python using pandas. Then I cleaned the data — some values were missing, so I dropped those rows to make it easier to work with.

After that, I had to convert the text data into numbers because ML models can't handle text. So I used label encoding for columns like gender and education. I also grouped age into 3 types (like young, middle, old) to make things simpler. Then I selected the useful columns to train the model.

Next, I split the data into training and testing, so I can train the model on one part and check accuracy on the other. I also scaled the number columns like income and credit score, because their values were too big compared to others, and models work better when all features are in similar range.

I trained a Random Forest Classifier model because it’s a good choice for binary classification like yes/no predictions. After training the model, I tested it and checked accuracy using the actual values. The model worked fine and showed decent accuracy on the test data.

After training the model, I wanted to see if it was being unfair to any group. For that I checked how many men and how many women got approved in the original data. I simply calculated the average approval for each gender. If there’s a big gap, then maybe the model or even the original data has some bias.

In my case, I just printed the difference in average approval for male and female. If the number is positive, it means men got approved more. If it’s negative, women got more approvals. This is a very basic check but gives some idea about fairness.

Overall, the project helped me understand how to take real-world data, clean it, prepare it for ML, train a model, and even think about fairness. The code was not perfect and I kept it simple on purpose. I didn’t use too many libraries or advanced features because I wanted to understand each step clearly.

In future I can try more models like XGBoost or add some graphs. Also maybe use AIF360 or other fairness tools to measure bias in more detail. But for now this was a good starting point to learn ML with ethics.
