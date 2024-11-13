import sys
import json
import os
import pandas
import pickle
from keras.models import load_model
import numpy as np

class ModelManager():
    def __init__(self):
        self.dirpath = os.path.dirname(os.path.realpath(__file__))

        self.cols = [
            "Sex",
            "Age",
            "Weight",
            "IsSmoking",
            "MaxDNSeverityCategory",
            "MinAbsLymph",
            "MaxAbsLeic",
            "MaxPlt",
            "MaxESR",
            "MaxCProtein",
            "MaxFerritin",
            "MaxDDimer",
            "MaxUrea",
            "MaxCommonProtein",
            "MaxGlucose",
            "MaxALT",
            "MaxAST",
            "MaxBilirubin",
            "MaxMNO",
            "MinProtrombIndex",
            "MaxFibrinogen",
            "MaxCreatinine",
            "MinHemoglobin",
            "MaxTemp",
            "MinSaturation",
            "MaxBP",
            "HasIBS",
            "HasMyocardInfarct",
            "HasONMK",
            "HasHypertonia",
            "HasHOBL",
            "HasDiabetes",
            "HasObesity",
            "HasHPN",
            "HasCancer",
            "HasHIV",
            "HasPneumo",
            "MaxKT",
            "HasAsthma",
            "CovidVac",
            "FluVac",
            "PneumococcusVac",
            "WasInResuscitation",
            "WasOnIVL"
        ]

        self.category_cols = ['Sex', 'IsSmoking', 'HasIBS', 'HasMyocardInfarct', 'HasONMK', 'HasHypertonia', 'HasHOBL',
                         'HasDiabetes', 'HasObesity', 'HasHPN', 'HasCancer', 'HasHIV', 'HasPneumo', 'MaxKT',
                         'HasAsthma',
                         'CovidVac', 'FluVac', 'PneumococcusVac', 'WasInResuscitation', 'WasOnIVL']

    def load_object(self, filename):
        filename = self.dirpath + '/bin/' + filename
        object = None
        with open(filename, 'rb') as f:
            object = pandas.read_pickle(f)
        return object

    def load_ml_model(self, modelname):
        filename = self.dirpath + '/models/' + modelname + "/" + modelname + ".pickle"
        model = None
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def encode_categorical(self, dataframe):
        for col in self.category_cols:
            dataframe[col] = dataframe[col].astype('bool')

        self.category_cols.append('MaxDNSeverityCategory')
        data_category_cols = dataframe[self.category_cols]
        data_category_cols['MaxDNSeverityCategory'] = data_category_cols['MaxDNSeverityCategory'].astype('string')

        data_category_cols = pandas.get_dummies(data_category_cols, dtype='float64')
        dummies_df = self.load_object('dummies.pickle')
        data_category_cols = data_category_cols.reindex(columns=dummies_df.columns, fill_value=0).drop(
            columns=['IsDead'])

        return data_category_cols

    def normalize_numeric(self, dataframe):
        numeric_cols = []
        for col in self.cols:
            if col not in self.category_cols:
                numeric_cols.append(col)

        data_numeric_cols = dataframe[numeric_cols]
        scaler = self.load_object('scaler.pickle')

        data_numeric_cols = pandas.DataFrame(
            scaler.transform(data_numeric_cols),
            index=data_numeric_cols.index,
            columns=data_numeric_cols.columns
        )

        return data_numeric_cols

    def evaluate(self, hasEmptyData, inputData):
        data = pandas.DataFrame(inputData)

        for col in self.cols:
            data[col] = data[col].astype(float)

        category_data = self.encode_categorical(data)
        numeric_data = self.normalize_numeric(data)

        data_processed = category_data.join(numeric_data)
        data_processed = data_processed.astype(float)

        models = [
            'covidNet',
            'histgboost'
        ]
        predictions = []
        for model_name in models:
            if model_name == 'covidNet' and not hasEmptyData:
                model = load_model(self.dirpath + "/models/covidnet/covidnet.h5")
                prediction = model.predict(data_processed)
                predictions.append({"model": model_name, "pred": prediction[:, 0].tolist()})
            else:
                if model_name != 'histgboost' and hasEmptyData:
                    continue
                model = self.load_ml_model(model_name)
                prediction = model.predict_proba(data_processed)
                predictions.append({"model": model_name, "pred": prediction[:, 1].tolist()})

        return predictions