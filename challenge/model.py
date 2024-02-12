import pandas as pd
from datetime import datetime
import numpy as np
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def is_high_season(self, fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def get_period_day(self, date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
        
    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = pd.read_csv('data/data.csv')
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        return data
    
    def xgboost_model(self, x_train, y_train):
        model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        model.fit(x_train, y_train)
        return model

    def lregression_model(self, x_train, y_train):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        return model
    
    def evaluate(self, model, x_test, which="lregression"):
        pred = model.predict(x_test)
        if which == "xgboost":
            pred = [1 if y_pred > 0.5 else 0 for y_pred in pred]
        return pred
        

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        test_features:pd.DataFrame,
        test_target:pd.DataFrame
        #test_size: None
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        #x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state = 42)
        xgboost_model = self.xgboost_model(features, target)
        lregression_model = self.lregression_model(features, target)

        xgboost_evaluate = self.evaluate(xgboost_model, test_features, "xgboost")
        lregression_evaluate = self.evaluate(lregression_model, test_features)
        
        recall_xgboost = recall_score(test_target, xgboost_evaluate)
        recall_lregression = recall_score(test_target, lregression_evaluate)

        if recall_lregression > recall_xgboost:
            self._model = lregression_model
        elif recall_lregression < recall_xgboost:
            self._model = xgboost_model
        else:
            self._model = xgboost_model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        pred = self._model.predict(features)
        return pred

if __name__ == "__main__":
    data = pd.read_csv("data/data.csv")
    delayModel = DelayModel()
    data = delayModel.preprocess(data)

    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
    )
    target = data['delay']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    delayModel.fit(X_train, y_train, X_test, y_test)
    pred = delayModel.predict(X_val)