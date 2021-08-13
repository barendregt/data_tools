"""
The ``data_tools.utils`` module provides a collection of utility functions for building machine learning models
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np


def add_prefix_to_column_names(df):
    """
    Transform dataframe column names by adding a prefix and removing numerical suffixes

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to transform. Must contain prefixes as column-names surrounded by single quotes (')

    Returns
    -------
    df: pandas.Dataframe
        Dataframe with transformed column names and without the prefix columns
    """

    df = df.copy()

    # find indices of "Table" names (these are surrounded by single quotes)
    table_indices = [i for i, s in enumerate(df.columns.values) if "'" in s]
    # remove single quotes around "Table" names before saving table names for later
    df.columns = df.columns.str.replace("'", "")
    table_cols = df.columns.values[table_indices]

    # create list of lists with start and end positions of columns to be renamed per table name
    table_indices.append(len(df.columns))
    rename_range = []

    for i in range(len(table_indices)-1):
        # start pos one after table position, end pos is next table position
        rename_range.append([table_indices[i]+1, table_indices[i+1]])

    # create list of new column names with which to override old column names
    new_cols = df.columns.values
    for prefix, lb, ub in np.hstack([table_cols[:, np.newaxis], np.array(rename_range)]):
        new_cols[lb:ub] = prefix+'_'+new_cols[lb:ub]

    new_cols = np.array([re.sub(r"\_\d*$", r"", c) for c in new_cols])

    # replace column names with new column names and drop table columns
    df.columns = new_cols
    df.drop(df.columns[table_indices[:-1]], axis=1, inplace=True)

    return df


def compute_time_difference(df, colpairs, time_unit='minutes', time_amount=1, append_result=True):
    """
    Computes the difference between one or more sets of datetime columns. Allows the unit and resolution of the result to be set.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe with all columns specified in **colpairs**
    colpairs: dict of list
        A dictionary of list of column-name pairs. The Key will be the column name for the result. The difference will be computed in the order that columns appear in the Value.
    time_unit: string, optional
        The unit of time for the result. Should be a string specifying a time unit supported by ``pandas.Timedelta``
    time_amount: int, optional
        The time resolution for the result. The difference will be computed as a multiple of **time_unit**
    append_result: bool, optional
        When set to ``True`` the original Dataframe will be returned with the new columns appended. Otherwise only returns new columns.

    Returns
    -------
    df: pandas.Dataframe
        Dataframe with the resulting differences

    Examples
    --------
    >>> df = pd.DataFrame([['2010-01-01','2010-01-03'],
    ...                 ['2010-01-01','2010-02-01']],
    ...                 columns=['firstDate','secondDate'],dtype='datetime64[ns]')
    >>> df
       firstDate secondDate
    0 2010-01-01 2010-01-03
    1 2010-01-01 2010-02-01
    >>> diff_dict = {'dateDiff':['firstDate','secondDate']}
    >>> compute_time_difference(df, diff_dict)
       firstDate secondDate  date_diff
    0 2010-01-01 2010-01-03    -2880.0
    1 2010-01-01 2010-02-01   -44640.0

    Get the difference in days rather than minutes and only output result columns

    >>> compute_time_difference(df, diff_dict, time_unit='days', append_result=False)
       date_diff
    0 -2.0
    1 -31.0

    Changing the order of the columns will change the result

    >>> diff_dict = {'dateDiff':['secondDate','firstDate']}
    >>> compute_time_difference(df, diff_dict, time_unit='days')
       firstDate secondDate  date_diff
    0 2010-01-01 2010-01-03       2.0
    1 2010-01-01 2010-02-01      31.0

    Change the output unit resolution to multiples of 4 hours

    >>> compute_time_difference(df, diff_dict, time_unit='hours', time_amount = 4)
       firstDate secondDate  date_diff
    0 2010-01-01 2010-01-03       12.0
    1 2010-01-01 2010-02-01      186.0
    """

    # Either append the results to the original Dataframe or store them in a new one
    if append_result:
        outdf = df
    else:
        outdf = pd.DataFrame([])

    for newcol, (col1, col2) in colpairs.items():
        exec(
            'outdf[newcol] = (df[col1]-df[col2])/ pd.Timedelta({}={})'.format(time_unit, time_amount))

    return outdf


def clean_ids(df, cols):
    """
    Force ID column values to string type and remove decimals (if present)

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe containing all columns specified in **cols**
    cols: list
        List of column names for which the values need to be converted

    Returns
    -------
    df: pandas.Dataframe
        Dataframe with transformed values for specified columns
    """
    for col in cols:
        df[col] = df[col].astype(float).astype(str).str.replace(r'\.0', '')

    return df


def print_shape(df, suffix_string=''):
    """
    Print shape of a dataframe in easy to read format with optional suffix

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to print shape of
    suffix_string: string, optional
        Additional text to print after the shape of the Dataframe

    """
    print('df has {:,} rows and {:,} columns {}'.format(
        df.shape[0], df.shape[1], suffix_string))


def print_nuniques_per_column(df, cols=None, return_result=False):
    """
    Print # of unique elements per column of a dataframe

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to examine
    cols: list, optional
        Limit to this subset of columns. Defaults to all columns if not specified.
    return_result: bool, optional
        Returns the result as a list in addition to printing the output.

    Returns
    -------
    unique_counts: list, optional
        Optionally returns the counts

    """

    # Default to using all columns if no subset is specified
    if cols is None:
        cols = df.columns.values

    max_len = max([len(col) for col in cols])
    df_len = len('{:,}'.format(df.shape[0]))

    unique_counts = []
    for col in cols:
        nunique = df[col].nunique()
        unique_counts.append(nunique)
        print('{:>{}}: {:>{},} unique values'.format(
            col, max_len, nunique, df_len))

    if return_result:
        return unique_counts


def print_null_percentage(df, print_all=False, threshold=0.5):
    """
    Print percentage of null values in all/sparse columns

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to examine
    print_all: bool, optional
        Print all columns or only columns that have a high (higher than **threshold**) percentage of null values.
    threshold: float, optional
        Set threshold for filtering out columns when **print_all** is set to ``False``. Default to ``0.5``
    """

    print("percentage null values per column:")
    for column in df.columns.values:
        percentage = df[column].isnull().sum()/len(df)
        if print_all == True:
            print("{:.4f}".format(percentage), column)
        else:
            if percentage > threshold:
                print("{:.4f}".format(percentage), column)


def filter_empty_rows(df, cols):
    """
    Filter empty rows from dataframe

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to examine
    cols: list
        List of columns to check for emptiness

    Returns
    -------
    df: pandas.DataFrame
        Input dataframe with the empty rows removed
    """
    for col in cols:
        df = df[df[col].notnull()]

    return df


def force_datetime_datatype(df, cols):
    """
    Force columns to datetime datatype, errors will result in NaT values.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to examine
    cols: list
        List of columns to convert

    Returns
    -------
    df: pandas.DataFrame
        Input dataframe with columns converted to datetime
    """
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def split_train_test_datetime(df, date_col, start_test, end_test,
                                            start_train=None, end_train=None,
                                            start_valid=None, end_valid=None,
                                            date_format='%d-%m-%Y', target_col=None):
    """
    Returns a train and test dataset from the source data based on a datetime range rather than number of samples or ratios. Can optionally return validation data.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataset to be split
    date_col: string
        Name of column to base the split on
    start_test: string
        Start date of the test data, formatted according to date_format
    end_test: string
        End date of the test data, formatted according to date_format
    start_train: string, optional
        Start date of the training data, formatted according to date_format. Defaults to everything outside the test data range.
    end_train: string, optional
        End date of the training data, formatted according to date_format. Defaults to everything outside the test data range.
    start_valid: string, optional
        Start date of the validation data, formatted according to date_format.
    end_valid: string, optional
        End date of the validation data, formatted according to date_format.
    date_format: string, optional
        Format string for the supplied dates (default = '%d-%m-%Y')
    target_col: string, optional
        Name of the target column (if present)

    Returns
    -------
    train_data, test_data, validation_data : pandas.Dataframe
        Extracted train, test and (optional) validation data
    train_target, test_target, validation_target: pandas.Series, optional
        If target_col is given, returns the corresponding train,test and (optional) validation target data
    """

    # Convert the supplied date ranges to Pandas datetime types
    start_test_date = pd.to_datetime(
        start_test, format=date_format, errors='coerce')
    end_test_date = pd.to_datetime(
        end_test, format=date_format, errors='coerce')

    if (start_train is not None) & (end_train is not None):
        start_train_date = pd.to_datetime(
            start_train, format=date_format, errors='coerce')
        end_train_date = pd.to_datetime(
            end_train, format=date_format, errors='coerce')

    if (start_valid is not None) & (end_valid is not None):
        start_valid_date = pd.to_datetime(
            start_valid, format=date_format, errors='coerce')
        end_valid_date = pd.to_datetime(
            end_valid, format=date_format, errors='coerce')

    # Sort the data by date for faster indexing
    df = df.sort_values(date_col)

    # Extract test data based on the supplied date range
    test = df[(pd.to_datetime(df[date_col]) >= start_test_date) &
              (pd.to_datetime(df[date_col]) <= end_test_date)]

    # Extract validation data if specified
    if (start_valid is not None) & (end_valid is not None):
        valid = df[(pd.to_datetime(df[date_col]) >= start_valid_date) &
                   (pd.to_datetime(df[date_col]) <= end_valid_date)]

    # Extract train data either based on a supplied date range
    # OR as everything that is not in the test data range
    if (start_train is not None) & (end_train is not None):
        train = df[(pd.to_datetime(df[date_col]) >= start_train_date) &
                   (pd.to_datetime(df[date_col]) <= end_train_date)]
    else:
        if (start_valid is not None) & (end_valid is not None):
            train = df[~(
                        ((pd.to_datetime(df[date_col]) >= start_test_date) &
                         (pd.to_datetime(df[date_col]) <= end_test_date)) |
                        ((pd.to_datetime(df[date_col]) >= start_valid_date) &
                         (pd.to_datetime(df[date_col]) <= end_valid_date))
                        )]
        else:
            train = df[~((pd.to_datetime(df[date_col]) >= start_test_date) &
                         (pd.to_datetime(df[date_col]) <= end_test_date))]

    # Collect and organize all the outputs
    outputs = []

    if target_col is not None:

        outputs.append(train.drop(target_col, axis=1))
        outputs.append(test.drop(target_col, axis=1))
        if (start_valid is not None) & (end_valid is not None):
            outputs.append(valid.drop(target_col, axis=1))

        outputs.append(train[target_col])
        outputs.append(test[target_col])
        if (start_valid is not None) & (end_valid is not None):
            outputs.append(valid[target_col])
    else:
        outputs.append(train)
        outputs.append(test)
        if (start_valid is not None) & (end_valid is not None):
            outputs.append(valid)

    return tuple(outputs)


def compute_cost_metric(y_true: np.array, y_pred: np.array, cost_matrix: np.array = None, labels=None, normalize: bool = False) -> np.array:
    """
    This computes the cost of errors made in the predictions. The calculation is a based on a provided cost matrix which should
    be an NxN matrix (N = number of classes) specifying the cost for each error combination.

    :param y_true: target labels
    :param y_pred: predicted labels
    :param cost_matrix: NxN matrix where N is number of classes in target
    :param normalize: normalize the result
    :return:
        cost of errors
    """

    if cost_matrix is None:
        print('Error: no cost matrix was supplied!')
        return np.NaN
    cm = np.transpose(confusion_matrix(y_true, y_pred, labels=labels))
    cost = sum(sum(cm * cost_matrix))
    if normalize:
        cost /= y_true.shape[0]
    return cost


def eval_model_performance(y_true, y_pred, verbose=False):
    """
    This function evaluates a classification model using a number of metrics: matthews correlation coefficient,
    f1 score and accuracy score.

    Parameters
    ----------
    y_true : pandas.Series, numpy.array, list
        Target labels
    y_pred : pandas.Series, numpy.array, list
        Predictions made by model
    verbose : boolean, optional
        If set to True will also print out the full classification report
    """

    # Determine averaging strategy for F1 score
    strat_add = ''
    if len(y_true.unique()) > 2:
        avg_strat = 'macro'
    else:
        avg_strat = 'binary'
        strat_add = ' (pos class)'

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1score = f1_score(y_true=y_true, y_pred=y_pred,
                       pos_label=y_true.unique()[1], average=avg_strat)
    mc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    if verbose:
        print('************************* Model performance **************************************')
        print('\t\t\t Accuracy:\t {:.2f}%'.format(accuracy * 100))

        print('\t\t\t F1 score:\t {:.2f} {}'.format(f1score, strat_add))
        print('\t\t\t Matthews corr:\t {:.2f}'.format(mc))
        print('**********************************************************************************')
        print('************************* Classification report **********************************')
        print(classification_report(y_true, y_pred))
        print('**********************************************************************************')
    else:
        print('[model performance] corr = {}, f1 = {}, accuracy = {}'.format(
            mc, f1score, accuracy))

    return accuracy, f1score, mc


def reconstruct_feature_names(fu):
  import sklearn
  # Define some internal helper functions

  def _reconstruct_categoricals(_fu):
    ohe_cats = []

    cat_dict = _fu.named_steps['indexer'].dictionaries

    for _feature in _fu.named_steps['selector'].columns:
        try:
            for key, _ in sorted(list(cat_dict[_feature].items()), key=lambda tup: tup[1]):
                ohe_cats.append('{}_{}'.format(_feature, key))
        except:
            ohe_cats.append(_feature)
    return ohe_cats

  def _reconstruct_woes(_fu):
    if len(_fu.named_steps['woe'].classes) == 2:
      woe_cols = [item for sublist in
                  [list(map(lambda x: 'woe_'+str(x),
                        _fu.named_steps['selector'].columns))]
                  for item in sublist]
    else:
        woe_cols = [item for sublist in
                    [list(map(lambda x: 'woe_'+str(x)+'_'+str(_class), _fu.named_steps['selector'].columns))
                    for _class in _fu.named_steps['woe'].classes]
                    for item in sublist]
    return woe_cols

  def _reconstruct_other(_fu):
    return _fu.named_steps['selector'].columns

  if not isinstance(fu, sklearn.pipeline.FeatureUnion):
    print('Error: Input needs to be an instance of sklearn.pipeline.FeatureUnion')
    return None

  fu = fu.transformer_list

  # Build list of augmented feature names
  named_features = []
  for ii, cat in enumerate([_c[0] for _c in fu]):
    if cat in ['woe', 'woes', 'weightofevidence']:
        cat = 'woes'
    elif cat in ['categorical', 'cat', 'cats', 'categoricals']:
        cat = 'categoricals'
    else:
        cat = 'other'
    
    named_features.extend(eval('_reconstruct_{}(fu[{}][1])'.format(cat, ii)))
  
  return named_features

def get_feature_importances(mdl, importance_type='gain', named_features=None):
  import pandas as pd
  fi = pd.DataFrame([])

  # Use internal names if no names were given
  fi['internal_names'] = mdl.get_booster().feature_names    
  if named_features is not None:
    fi['Feature'] = named_features
  else:
    fi['Feature'] = fi['internal_names']

  fi = pd.merge(fi, pd.DataFrame.from_dict(mdl.get_booster().get_score(importance_type=importance_type), orient='index', columns=[importance_type]), left_on='internal_names', right_index=True, how='left').fillna(0)    

  return fi

def log_mlflow(model, results, metrics = {}, params = {}, run_id = None, experiment_id = None):
    """
    Log the results of a model train run to MLflow.

    Parameters
    ----------
    model : scikit-learn estimator or Pipeline object
        The trained model object (needs to be pickleable)
    results : list or tuple
        A set of lists of predictions and corresponding targets. This can either be just 
        for the test data or also include results on the training data
    metrics : dict, optional
        A set of key-value pairs describing the metrics to log. The key should be a string naming the metric and 
        the value a callable with (at least) the arguments y_true (targets) and y_pred (predictions) that returns 
        a singe scalar metric as output.
    params : dict, optional
        Optional parameters that can be logged with the model run. This dict will be passed as is to the mlflow.log_params() function,
        so the requirements are the same as that function.
    run_id : string, optional
        If specified, get the run with the specified UUID and log parameters and metrics under that run. The run’s end time is unset and 
        its status is set to running, but the run’s other attributes (source_version, source_type, etc.) are not changed.
    experiment_id : string, optional
        ID of the experiment under which to create the current run (applicable only when run_id is not specified). If experiment_id 
        argument is unspecified, will look for valid experiment in the following order: activated using set_experiment, 
        MLFLOW_EXPERIMENT_NAME environment variable, MLFLOW_EXPERIMENT_ID environment variable, or the default experiment as 
        defined by the tracking server.        
    """

    include_train_results = False

    if len(results) == 4:
        train_preds, train_targets, test_preds, test_targets = results
        include_train_results = True
    else:
        test_preds, test_targets = results

    with mlflow.start_run(run_id = run_id, experiment_id = experiment_id):
        if len(metrics) > 0:
            if include_train_results:
                # Log metrics for training data
                [mlflow.log_metric(_k + '_train', _v(y_true=train_targets, y_pred=train_preds)) for _k, _v in metrics.items()]

            # Log metrics for test data
            [mlflow.log_metric(_k + '_test', _v(y_true=test_targets, y_pred=test_preds)) for _k, _v in metrics.items()]

        # Log any parameters that were given
        if len(params) > 0:
            mlflow.log_params(params)
            
        # Store model artifact
        mlflow.sklearn.log_model(model, 'model')  
