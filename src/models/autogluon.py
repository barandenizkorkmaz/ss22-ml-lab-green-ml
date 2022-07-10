from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime
from autogluon.core.metrics import make_scorer
from metrics import rmspe

subsets = ['pretrained']
target_layers = ['dense']

for subset in subsets:
    for target_layer in target_layers:

        train_data_path = f"/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/layerwise-{subset}-{target_layer}-train"
        test_data_path = f"/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/layerwise-{subset}-{target_layer}-test"

        now = datetime.now() # current date and time
        timestamp = now.strftime("%m-%d-%Y--%H:%M:%S")

        time_limit = 600
        presets='best_quality'

        train_data = TabularDataset(train_data_path)
        train_data.head()

        label = 'energy_consumption'
        print("Summary of class variable: \n", train_data[label].describe())

        save_path = f'agModels-greenML-{timestamp}-{subset}-{target_layer}'  # specifies folder to store trained models
        predictor = TabularPredictor(label=label, path=save_path).fit(train_data, time_limit=time_limit, presets=presets)

        test_data = TabularDataset(test_data_path)
        y_test = test_data[label]  # values to predict
        test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
        test_data_nolab.head()

        y_pred = predictor.predict(test_data_nolab)
        print("Predictions:  \n", y_pred)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

        ag_rmspe_custom_scorer = make_scorer(
            name='rmspe',
            score_func=rmspe,
            optimum=0,
            greater_is_better=False)

        predictor.leaderboard(test_data, extra_metrics=[ag_rmspe_custom_scorer]).to_csv(f"{timestamp}-results-{subset}-{target_layer}.csv", encoding='utf-8', index=False)

        y_pred_np = y_pred.to_numpy().flatten()
        y_test_np = y_test.to_numpy().flatten()