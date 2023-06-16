# CS3308-ML-Project-Retrosynthetic-Planning
Code for CS3308 ML Project: Retrosynthetic Planning
* This is code for CS3308 ML Project: Retrosynthetic Planning
* The project is composed of three parts. For the task1, we raise two models(SVM/MLP) to predict the template for the given product; for the task2, we raise three model(MLP,KNN,Adaboost based on dicisiontree) to predict the cost for the given product, for the test3, we use retro algorithm for Multi-step Retrosynthetic planning.
## Task 1
* training and test data are stored in data directory, can use `preprocessing.py` for data generation
* log directory for training and test outputs record
* model directory for trained model storage
* The src directory stores the source files:
  1. For training, you can just use 'python single-step.py > ...' for model training and get the training output, there are SVM and MLP model supported and parameters can be changed.
  2. For testing, you can just use 'python test.py > ...' for model test and get the test output. Be care to load the right model in model directory.
## Task 2
* training and test data are stored in data directory, no need to generate data
* log directory for training and test outputs record
* model directory for trained model storage
* The src directory stores the source files:
  1. For training, use 'python mole_evl.py > ...' for model training and get the training output, there are MLP, KNN-regressor and Adaboost model supported and parameters can be changed.
  2. For testing, you can just use 'python test.py > ...' for model test and get the test output. Be care to load the right model in model directory. Care that test function for mlp is 'Test_MLP(test_data)' which is different from other model
## Task 3
* In this task, retro algorithm provided by ![link](https://github.com/binghong-ml/retro_star) is learned and the demanded environment is shown 'environment.yml'
* You can used the model trained in the former part for task 3 test by putting them in the 'one_step_model' and 'saved_models' respectively and change the parameters in 'args'.
