# StreamlitDeployment

The live link for Part 2 : https://jix99-streamlitdeployment-nxtlabdeploy-model-30ihji.streamlitapp.com/

# Part 1

1.Write a regex to extract all the numbers with orange color background from the below text in italics.
 {"orders":[{"id":1},{"id":2},{"id":3},{"id":4},{"id":5},{"id":6},{"id":7},{"id":8},{"id":9},{"id":10},{"id":11},{"id":648},{"id":649},{"id":650},{"id":651},{"id":652},   {"id":653}],"errors":[{"code":3,"message":"[PHP Warning #2] count(): Parameter must be an array or an object that implements Countable (153)"}]}


# Part 2
 
1. Problem Statement: Train a machine learning model (preferably with a neural network) that predicts the customer who is going to be checked in.
2. Executed Exploratory Data Analysis on the data to understand the patterns and anamolies. (eda.ipynb)
3. Preprocessed the data so that it can be ready to train the model. (preprocessing.ipynb)
4. Vectorised the data in order to make the data to feed it to the model so that it can get trained and validated. (preprocessing.ipynb)
5. Saved the weights of the trained model and used it in the final application py file (deploy_model.ipynb)
6. Built a complete working python pipeline which takes input as test_data and sends it to the trained Deep Learning Model, which later predicts the output.(deploy_model.py)
7. I used streamlit to build the user interface application.
8. In order to deploy the application the basic things needed are 
      - .py file
      - requirements.txt
      - setup.sh
      - Procfile
      - Data
9. Uploaded the file in github.
10. Login to streamlit cloud app and link the cloud with the github file and deploy the application, for more information refer: https://carpentries-incubator.github.io/python-interactive-data-visualizations/08-publish-your-app/index.html

1. Write about any difficult problem that you solved. (According to us, difficult - is something which 90% of people would have only 10% probability in getting a similarly good solution). 

The most difficult problem I encountered was in my engineering career where I had enrolled in a hackathon conducted by my college. As an electronic engineer I was given a project to build a car which had to overcome several obstacles and we built the car using several sensors in order to detect and avoid the obstacles. Everything was going smooth until we found one major flaw in our design that our car was using an proximity sensor to detect the obstacle which worked perfectly for all the obstacles in a straight lane but the car had to climb an inclined plane at one part of the course, so when the car approached near the inclined plane, the car suddenly stopped because the car was interpreting the inclined plane as the obstacle so our whole aim was about to shatter, so as a team we all came up with different ideas to overcome it but none of it worked but at last I came with the plan to deploy two proximity sensors, one at the bottom and other at the top and configured it to hover through the inclined plane successfully.

As far as the Data Science field is concerned, I have faced all the problems usually faced by beginners, as one has to transition from basic python programming to complex programming in ML and DL. After encountering errors it is very important to smartly search to tackle those errors and learn from the documentations.


2. Explain back propagation and tell us how you handle a dataset if 4 out of 30 parameters have null values more than 40 percentage

Back propagation:

Back propagation is the fundamental part in any DL model which revolutionised the AI and DS industry. It is a part of the process of training the model alongside forward propagation. The backbone of any back propagation process is the chain rule of differentiation, with it one can update the weights and biases in order to reduce the loss function in most cases. So in order for backpropagation to work the function used in the model should be differentiable and continuous. If it satisfies the above condition and applied properly there will be many wonders and less blunders.

To handle a dataset if 4 out of 30 parameters have null values more than 40 percentage we can follow either of the following ways:
 
 -To delete the rows which has the missing values but this can lead to serious loss in the amount of data and this should be the last option one should keep in choice
 
 -To replace the null values with mean, median or mode which tries to maintain the distribution of the data,  without much disturbing it but it can cause data leakage problems.
 
 -To assign a unique category to the missing values, This strategy will add more information into the dataset which will result in the change of variance. 
 
 -Predicting the missing values using the features which do not have missing values, we can predict the nulls with the help of a machine learning algorithm. This       method may result in better accuracy, unless a missing value is expected to have a very high variance.
