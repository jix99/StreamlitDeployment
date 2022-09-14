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
