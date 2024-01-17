import os
from datetime import datetime
import glob
import dill
import pandas as pd
import json
from sklearn.pipeline import Pipeline

# path = os.environ.get('PROJECT_PATH', '../airflow_hw')
path = os.path.expanduser('~/airflow_hw')

def predict(): #->pd.DataFrame:

    mod = sorted(os.listdir(f'{path}/data/models'))# сортируем модели по дате создания в указанной папке

    with open(f'{path}/data/models/{mod[-1]}', 'rb') as file:#mod[-1] самая свежая по дате модель
        model = dill.load(file)

    preds = pd.DataFrame(columns=['car_id', 'pred'])

    for file in glob.glob(f'{path}/data/test/*.json'):#перебираем файлы которые будем предсказывать
        with open(file) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            preds = pd.concat([preds, df1], axis = 0)
    print(preds)

    preds.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
