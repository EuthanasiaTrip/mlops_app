import sys
import json
import os
import pandas
import pickle
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Dense
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from datetime import datetime

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
# from evidently.test_suite import TestSuite
# from evidently.test_preset import DataQuality, DataStability
from evidently.tests import *

from . import pgconnector as pgconnector

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
            "WasOnIVL",
            "IsDead"
        ]

        self.category_cols = ['Sex', 'IsSmoking', 'HasIBS', 'HasMyocardInfarct', 'HasONMK', 'HasHypertonia', 'HasHOBL',
                         'HasDiabetes', 'HasObesity', 'HasHPN', 'HasCancer', 'HasHIV', 'HasPneumo', 'MaxKT',
                         'HasAsthma',
                         'CovidVac', 'FluVac', 'PneumococcusVac', 'WasInResuscitation', 'WasOnIVL', 'IsDead', 'MaxDNSeverityCategory']

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

    def encode_categorical(self, dataframe, isPreload=True):
        for col in self.category_cols:
            if col != 'MaxDNSeverityCategory' and col != "IsDead":
                dataframe[col] = dataframe[col].astype('bool')

        data_category_cols = dataframe[self.category_cols]
        data_category_cols['MaxDNSeverityCategory'] = data_category_cols['MaxDNSeverityCategory'].astype('string')

        data_category_cols = pandas.get_dummies(data_category_cols, dtype='float64')
        if isPreload:
            dummies_df = self.load_object('dummies.pickle')
            data_category_cols = data_category_cols.reindex(columns=dummies_df.columns, fill_value=0).drop(
                columns=['IsDead'])

        return data_category_cols

    def normalize_numeric(self, dataframe, isPreload=True):
        numeric_cols = []
        for col in self.cols:
            if col not in self.category_cols:
                numeric_cols.append(col)

        data_numeric_cols = dataframe[numeric_cols]
        if isPreload:
            scaler = self.load_object('scaler.pickle')
        else:
            scaler = StandardScaler()
            scaler.fit(data_numeric_cols)

        data_numeric_cols = pandas.DataFrame(
            scaler.transform(data_numeric_cols),
            index=data_numeric_cols.index,
            columns=data_numeric_cols.columns
        )

        return data_numeric_cols
    
    def create_nn_model(self, shape):
        input_shape_numeric = (shape)
        model_input_numeric = Input(shape=input_shape_numeric)
        dense_1 = Dense(16, activation="relu")(model_input_numeric)
        dense_2 = Dense(64, activation="relu")(dense_1)
        dense_3 = Dense(32, activation="relu")(dense_2)
        dropout = Dropout(0.25)(dense_3)
        dense_4 = Dense(16, activation="relu")(dropout)
        out = Dense(1, activation="sigmoid")(dense_4)
        model = Model(model_input_numeric, out)
        loss = 'binary_crossentropy'
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        metrics='accuracy'
        model.compile(optimizer=optimizer,loss=loss,metrics=[metrics, tf.keras.metrics.BinaryIoU(name='io_u')])
        return model
    
        
    def generate_report(self, model):
        column_map = ColumnMapping()
        column_map.target = 'IsDead'
        column_map.prediction = 'pred'

        dbcon = pgconnector.DBConnector()
        train_data = pandas.DataFrame(dbcon.selectCols(self.cols), columns=self.cols)
        current_data = pandas.DataFrame(dbcon.selectCols(self.cols, tableName="PATIENTS_DATA"), columns=self.cols)

        train_data = self.preprocess_dataframe(train_data)
        current_data = self.preprocess_dataframe(current_data)

        print(train_data.columns.tolist())

        train_data['pred'] = model.predict(train_data.drop(columns='IsDead'))
        current_data['pred'] = model.predict(current_data.drop(columns='IsDead'))

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=train_data, current_data=current_data, column_mapping=column_map)
        timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        report_path = f'tests-{timestamp}.html'
        report.save_html(report_path)

        return open(report_path, 'r', encoding='utf-8')

    def preprocess_dataframe(self, df):
        for col in self.cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(float)
            if df[col].dtype == 'float':
                mode = df[col].mode()
                df.fillna({col: mode[0]}, inplace=True)    

        df['Sex'] = df['Sex'].astype(bool)
        df['CovidVac'] = df['CovidVac'].astype(bool)
        df['FluVac'] = df['FluVac'].astype(bool)
        df['PneumococcusVac'] = df['PneumococcusVac'].astype(bool)

        category_data = self.encode_categorical(df, isPreload=False)
        numeric_data = self.normalize_numeric(df, isPreload=False)

        data_processed = category_data.join(numeric_data)
        data_processed = data_processed.astype(float)

        return data_processed
    
    def train(self, num_epochs=25, return_report=False):
        dbcon = pgconnector.DBConnector()

        df = pandas.DataFrame(dbcon.selectCols(self.cols), columns=self.cols)

        data_processed = self.preprocess_dataframe(df)

        params_to_smote = data_processed.drop(columns='IsDead')
        labels_to_smote = data_processed['IsDead']

        oversampling = SMOTE()
        params_resample, labels_resample = oversampling.fit_resample(params_to_smote, labels_to_smote)
        params_resample["IsDead"] = labels_resample

        test_split = 0.2

        params = params_resample.drop(columns='IsDead')
        labels = params_resample['IsDead']

        print(params_to_smote.columns.tolist())

        parameters_train_df, parameters_testval_df, labels_train_df, labels_testval_df = train_test_split(
            params,
            labels,
            test_size=test_split,
            random_state=42,
            shuffle=True
        )

        parameters_val_df, parameters_test_df, labels_val_df, labels_test_df = train_test_split(
            parameters_testval_df,
            labels_testval_df,
            test_size=0.5,
            random_state=42,
            shuffle=True
        )

        model = self.create_nn_model(params.shape[1])

        model.fit(
            parameters_train_df,
            labels_train_df,
            validation_data=(
                parameters_val_df,  
                labels_val_df
                ),               
            epochs = num_epochs,
            batch_size=25
        )

        timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        model.save(self.dirpath + f'/models/fresh/covidnet-{timestamp}.h5')

        if return_report:
            return self.generate_report(model)
        else:
            loss, acc, mean_io_u = model.evaluate(parameters_test_df, labels_test_df)
            return acc

    def evaluate(self, hasEmptyData, inputData):
        data = pandas.DataFrame(inputData)

        for col in self.cols:
            if col != "IsDead":
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
