from Utils import *
from StudentModel import DKTModel, DataGenerator

# 定义常量
dataset = "data/ASSISTments_skill_builder_data.csv" # Dataset path
best_model_file = "saved_models/ASSISTments.best.model.weights.hdf5" # File to save the model.
train_log = "logs/dktmodel.train.log" # File to save the training log.
eval_log = "logs/dktmodel.eval.log" # File to save the testing log.
optimizer = "adagrad" # Optimizer to use
lstm_units = 250 # Number of LSTM units
batch_size = 20 # Batch size
epochs = 10 # Number of epochs to train
dropout_rate = 0.6 # Dropout rate
verbose = 1 # Verbose = {0,1,2}
testing_rate = 0.2 # Portion of data to be used for testing
validation_rate = 0.2 # Portion of training data to be used for validation


# 数据预处理
dataset, num_skills = read_file(dataset)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(dataset, validation_rate, testing_rate)

print("======== Data Summary ========")
print("Data size: %d" % len(dataset))
print("Training data size: %d" % len(X_train))
print("Validation data size: %d" % len(X_val))
print("Testing data size: %d" % len(X_test))
print("Number of skills: %d" % num_skills)
print("==============================")


# 构建数据迭代器 for training/testing/validation
train_gen = DataGenerator(X_train[0:10], y_train[0:10], num_skills, batch_size)
val_gen = DataGenerator(X_val[0:10], y_val[0:10], num_skills, batch_size)
test_gen = DataGenerator(X_test[0:10], y_test[0:10], num_skills, batch_size)

# 创建模型
student_model = DKTModel(num_skills=train_gen.num_skills,
                      num_features=train_gen.feature_dim,
                      optimizer=optimizer,
                      hidden_units=lstm_units,
                      batch_size=batch_size,
                      dropout_rate=dropout_rate)

# 训练模型
history = student_model.fit(train_gen,
                  epochs=epochs,
                  val_gen=val_gen,
                  verbose=verbose,
                  filepath_bestmodel=best_model_file,
                  filepath_log=train_log)


# 加载最优模型（Best Validation Loss）
student_model.load_weights(best_model_file)


# 测试模型
result = student_model.evaluate(test_gen, metrics=['auc','acc','pre'], verbose=verbose, filepath_log=eval_log)