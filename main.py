from models.ml_models import MLModels

ml = MLModels()
models = ml.build_basic_models(x_train, y_train)
models_acc, models_f1 = ml.evaluate_basic_models(x_test, y_test)
