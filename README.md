# udacity-disaster-project

#### 1. Project Topic
The application that receives a message and matches what kind of disaster it is


#### 2. Project Motivation
It's a good project to apply my data engineering and programming skills to analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.


#### 3. File Descriptions
- app directory
  - templates directory : This directory includes html files for layout of the web page
  - run.py : This python file executes flask web app
- data directory
  - csv files which are raw data
  - DisasterResponse.db : This database file has cleaned data after data cleansing
  - process_data.py : This python file is a ETL pipeline
     - Loads the messages and categories datasets
     - Merges the two datasets
     - Cleans the data
     - Stores it in a SQLite database
- models directory
  - classifier.pkl : The model which classifies what kind of disaster after accepting a message as an input
  - train_classifier.py : This python file is a ML pipeline
     - Loads data from the SQLite database
     - Splits the dataset into training and test sets
     - Builds a text processing and machine learning pipeline
     - Trains and tunes a model using GridSearchCV
     - Outputs results on the test set
     - Exports the final model as a pickle file


#### 4. How to execute this project
  1. Execute 2 python files (process_data.py, train_classifier.py) in terminal by typing two each lines as below:
  
    - python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    - python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    
  2. In order to run the web app,
  
    - Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file
    - Type in the command line: python run.py
    - In a new web browser window, type in the following: https://view6914b2f4-3001.udacity-student-workspaces.com


#### 5. How to Interact with this project
If something to want to fix, you can be a contributor of this project


#### 6. Creator
Baekspace
