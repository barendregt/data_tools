"""
The ``data_tools.transformers`` module provides a set of custom Transformers that can be used directly or as part of a scikit-learn pipeline
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class StringIndexer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dictionaries = dict()
        self.columns = list()

    def fit(self, X, y=None):
        self.columns = X.columns.values
        for col in self.columns:
            categories = np.unique(X[col])
            self.dictionaries[col] = dict(zip(categories, range(len(categories))))
        return self

    def transform(self, X):
        column_array = []
        for col in self.columns:
            dictionary = self.dictionaries[col]
            na_value = len(dictionary) + 1
            transformed_column = X[col].apply(lambda x: dictionary.get(x, na_value))
            column_array.append(transformed_column.values.reshape(-1, 1))
        return np.hstack(column_array)

class ModelPredictor(BaseEstimator, TransformerMixin):
    """
    Class to be used in sklearn pipeline, for a given model returns prediction in a column.
    Useful when prediction of a different model is useful input feature.

    Parameters
    ----------
    model : opend pickel file of model
        The actual model object.
    feature_name : string
        The name of the created column with predictions.
    dtype : string, must be an existing dtype
        The dtype to which to transform the predicted column to

    Returns
    -------
    output_df : pandas.Dataframe
        Returns the input pandas.Dataframe concatenated with a column containing the
        predictions of the model
    """
    def __init__(self, model, feature_name='model', dtype='str'):
        self.model = model
        self.feature_name = feature_name
        self.dtype = dtype

    def fit(self, X, y=None):
        """
        Doesn't do anything, for compatibility purposes only.

        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe that the model needs as input for prediction

        Returns
        -------
        output_df : pandas.Dataframe
            Returns the input dataframe unchanged
        """
        return self

    def transform(self, X, y=None):
        """
        Invokes predict method of model to create predictions. Attaches prediction to input
        dataframe and returns concatenated dataframe.

        Parameters
        ----------
        X : pandas.Dataframe
            The dataframe that the model needs as input for prediction.

        Returns
        -------
        output_df : pandas.Dataframe
            Returns the concatenated (input and prediction) dataframe.
        """
        X = X.reset_index(drop=True)
        preds = pd.Series(self.model.predict(X), name=self.feature_name).astype(self.dtype)
        return pd.concat([X, preds], axis=1)

class PdFeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, union):
        self.union = union

    def fit(self, X, y=None, **fit_params):
        return PdFeatureUnion([one.fit(X, y) for one in self.union])

    def transform(self, X):
        return pd.concat([one.transform(X) for one in self.union], axis=1, join_axes=[X.index])

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms a high-capacity categorical value
    into Weigh of Evidence scores. Can be used in sklearn pipelines.
    """

    def __init__(self, verbose=0, cols=None, return_df=True,
                 smooth=0.5, fillna=0, dependent_variable_values=None):
        """
        :param smooth: value for additive smoothing, to prevent divide by zero
        """
        # make sure cols is a list of strings
        if not isinstance(cols, list):
            cols = [cols]

        self.stat = {}
        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.smooth = smooth
        self.fillna = fillna
        self.dependent_variable_values = dependent_variable_values

        self.classes = []

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('Input should be an instance of pandas.DataFrame()')

        if self.dependent_variable_values is not None:
            y = self.dependent_variable_values

        self.classes = np.unique(y)

        df = X[self.cols].copy()
        y_col_index = len(df.columns) + 1
        df[y_col_index] = np.array(y)

        def get_totals(x):
            total = np.size(x)
            pos = max(float(np.sum(x)), self.smooth)
            neg = max(float(total - pos), self.smooth)
            return pos, neg

        # get the totals per class
        total_positive, total_negative = get_totals(y)
        if self.verbose:
            print("total positives {:.0f}, total negatives {:.0f}".format(total_positive, total_negative))

        def compute_bucket_woe(x):
            bucket_positive, bucket_negative = get_totals(x)
            return np.log(bucket_positive / bucket_negative)

        # compute WoE scores per bucket (category)
        stat = {}
        for col in self.cols:

            if self.verbose:
                print("computing weight of evidence for column {:s}".format(col))

            stat[col] = ((df.groupby(col)[y_col_index].agg(compute_bucket_woe)
                         + np.log(total_negative / total_positive)).to_dict())

        self.stat = stat

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('Input should be an instance of pandas.DataFrame()')

        df = X.copy()

        # join the WoE stats with the data
        for col in self.cols:

            if self.verbose:
                print("transforming categorical column {:s}".format(col))

            stat = pd.DataFrame.from_dict(self.stat[col], orient='index')

            ser = (pd.merge(df, stat, left_on=col, right_index=True, how='left')
                   .sort_index()
                   .reindex(df.index))[0]

            # fill missing values with
            if self.verbose:
                print("{:.0f} NaNs in transformed data".format(ser.isnull().sum()))
                print("{:.4f} mean weight of evidence".format(ser.mean()))

            df[col] = np.array(ser.fillna(self.fillna))

        if not self.return_df:
            out = np.array(df)
        else:
            out = df

        return out

class WeightOfEvidenceEncoderMultiClass(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms a high-capacity categorical value
    into Weigh of Evidence scores. Can be used in sklearn pipelines.
    """

    def __init__(self, verbose=0, cols=None, return_df=True,
                 smooth=0.5, fillna=0, dependent_variable_values=None):
        """
        :param smooth: value for additive smoothing, to prevent divide by zero
        """
        # make sure cols is a list of strings
        if not isinstance(cols, list):
            cols = [cols]

        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.smooth = smooth
        self.fillna = fillna
        self.dependent_variable_values = dependent_variable_values

        #keep an array of woe-encoders
        self.classes = []
        self.woe_encorders = []
    def fit(self, X, y):
        #make sure to overwrite self.woe_encorders = []
        self.woe_encorders = []

        self.classes = np.unique(y)
        for class_ in self.classes:
            woe_encoder = WeightOfEvidenceEncoder(verbose=self.verbose, cols=self.cols, return_df=True,
                 smooth=self.smooth, fillna=self.fillna, dependent_variable_values=self.dependent_variable_values)
            woe_encoder.fit(X,y==class_)
            self.woe_encorders = self.woe_encorders + [woe_encoder]
        return self

    def transform(self, X, y=None):
        #for woe_encoder in self.woe_encorders:
        df = X.copy()

        for woe_encoder,class_ in zip(self.woe_encorders,self.classes):
            temp_ = woe_encoder.transform(X)
            #change name cols to woe_cols_NAME CLASS
            new_name = list(map(lambda x: 'woe_'+str(x)+'_'+str(class_),self.cols))
            df[new_name] = temp_.loc[:,self.cols]

        if not self.return_df:
            out = np.array(df)
        else:
            out = df
        select_cols=list(set(out.columns)-set(self.cols))
        out = out.loc[:,select_cols]
        return out

class WeightOfEvidenceEncoderContinuous(BaseEstimator, TransformerMixin):
    """
    Feature-engineering class that transforms a high-capacity categorical value
    into Weigh of Evidence scores for a continuous target. Can be used in sklearn pipelines.
    Based on: https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """

    def __init__(self, verbose=0, cols=None, return_df=True,
                 smooth=0.5, fillna=0, dependent_variable_values=None):
        """
        :param smooth: value for additive smoothing, to prevent divide by zero
        """
        # make sure cols is a list of strings
        if not isinstance(cols, list):
            cols = [cols]

        self.stat = {}
        self.return_df = return_df
        self.verbose = verbose
        self.cols = cols
        self.classes = ['',''] # backward compatibility
        self.smooth = smooth
        self.fillna = fillna
        self.dependent_variable_values = dependent_variable_values

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('Input should be an instance of pandas.DataFrame()')

        if self.dependent_variable_values is not None:
            y = self.dependent_variable_values
        df = X[self.cols].copy()
        y_col_index = len(df.columns) + 1
        df[y_col_index] = np.array(y)

        # Compute overall average as proxy for prior
        baseline_average = np.nanmean(df[y_col_index].values)

        if self.verbose:
            print("overall average of target (prior): {:.2f}".format(baseline_average))

        # compute WoE scores per bucket (category)
        stat = {}
        for col in self.cols:

            if self.verbose:
                print("computing weight of evidence for column {:s}".format(col))

            agg_col = (df.groupby(col)[y_col_index].agg(['mean','count']) / np.array([1,df.shape[0]])).rename(columns={'count':'lambda'})
            stat[col] = ((agg_col['lambda'] * baseline_average) + ((1-agg_col['lambda'])*(agg_col['mean']))).to_dict()
        self.stat = stat

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError('Input should be an instance of pandas.DataFrame()')

        df = X.copy()

        # join the WoE stats with the data
        for col in self.cols:

            if self.verbose:
                print("transforming categorical column {:s}".format(col))

            stat = pd.DataFrame.from_dict(self.stat[col], orient='index')

            ser = (pd.merge(df, stat, left_on=col, right_index=True, how='left')
                   .sort_index()
                   .reindex(df.index))[0]

            # fill missing values with
            if self.verbose:
                print("{:.0f} NaNs in transformed data".format(ser.isnull().sum()))
                print("{:.4f} mean weight of evidence".format(ser.mean()))

            df[col] = np.array(ser.fillna(self.fillna))

        if not self.return_df:
            out = np.array(df)
        else:
            out = df

        return out