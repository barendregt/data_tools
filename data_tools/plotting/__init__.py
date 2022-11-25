"""
The ``data_tools.plotting`` module provides some useful plotting functions used for evaluating ML models
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from scipy.special import inv_boxcox
import itertools

sns.set(style="ticks")

ML_PLOT_GLOBAL_THEME = "light"

ML_PLOT_COLORS = {
    "dark": {
        "title": "w",
        "xlabel": "w",
        "ylabel": "w",
        "xticks": "w",
        "yticks": "w",
        "spines": "w",
        "text": "w",
        "legend": "w",
        "background": "k",
    },
    "atom-dark": {
        "title": "w",
        "xlabel": "w",
        "ylabel": "w",
        "xticks": "w",
        "yticks": "w",
        "spines": "w",
        "text": "w",
        "legend": "w",
        "background": "#282C34",
    },
    "light": {
        "title": "k",
        "xlabel": "k",
        "ylabel": "k",
        "xticks": "k",
        "yticks": "k",
        "spines": "k",
        "text": "k",
        "legend": "k",
        "background": "w",
    },
}

# Add Jupyter specific which is identical to Atom (since jupyter uses this theme)
ML_PLOT_COLORS["jupyter-dark"] = ML_PLOT_COLORS["atom-dark"]

ML_PLOT_FONTS = {
    "title": 24,
    "xlabel": 16,
    "ylabel": 16,
    "xticks": 12,
    "yticks": 12,
    "text": 14,
}


def catplot(
    data,
    x,
    y,
    hue=None,
    kind="bar",
    title=None,
    rotate_x_ticks=False,
    unit_fmt=None,
    unit_data=None,
    color_theme=ML_PLOT_GLOBAL_THEME,
    xlabel="",
    ylabel="",
    axis_limits=None,
    xticks_off=False,
    yticks_off=False,
    legend=False,
    legend_custom_txt=None,
    legend_location="upper right",
    legend_outside=False,
    **kwargs
):

    f = sns.catplot(data=data, x=x, y=y, hue=hue, kind=kind, legend=False, **kwargs)

    if rotate_x_ticks:
        plt.xticks(rotation=45, horizontalalignment="right")

    if title is not None:
        plt.title(
            title,
            fontsize=ML_PLOT_FONTS["title"],
            color=ML_PLOT_COLORS[color_theme]["title"],
        )

    if legend:

        if legend_custom_txt is not None:
            if legend_outside:
                leg = f.fig.legend(
                    legend_custom_txt,
                    loc=legend_location,
                    framealpha=0,
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                )
            else:
                leg = f.fig.legend(legend_custom_txt, loc=legend_location, framealpha=0)
        else:
            if legend_outside:
                leg = f.fig.legend(
                    loc=legend_location,
                    framealpha=0,
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                )
            else:
                leg = f.fig.legend(loc=legend_location, framealpha=0)

        for text in leg.get_texts():
            plt.setp(
                text,
                color=ML_PLOT_COLORS[color_theme]["legend"],
                fontsize=ML_PLOT_FONTS["text"],
            )

    if xticks_off:
        plt.xticks([])
    plt.xticks(
        color=ML_PLOT_COLORS[color_theme]["xticks"], fontsize=ML_PLOT_FONTS["xticks"]
    )

    if yticks_off:
        plt.yticks([])
    plt.yticks(
        color=ML_PLOT_COLORS[color_theme]["yticks"], fontsize=ML_PLOT_FONTS["yticks"]
    )

    plt.tick_params(
        axis="both",  # changes apply to both axes
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=True,
    )

    for _axii, _fax in enumerate(f.axes.flatten()):

        if (kind == "bar") & (unit_fmt is not None):
            for _barii, p in enumerate(_fax.patches):

                if (unit_data is None) | (len(unit_data) != len(_fax.patches)):
                    _h = p.get_height()
                else:
                    _h = unit_data[_barii]

                _x = p.get_x() + (p.get_width() / 2)
                # _fax.text(_x, _h + 0, unit_fmt.format(_h), horizontalalignment='center', verticalalignment='bottom', color=, fontsize=);
                _fax.annotate(
                    s=unit_fmt.format(_h),
                    xy=(_x, _h),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color=ML_PLOT_COLORS[color_theme]["text"],
                    weight="bold",
                    fontsize=ML_PLOT_FONTS["text"],
                )
        elif (kind == "point") & (unit_fmt is not None):
            lines = _fax.lines
            x_len = len(lines[0].get_xdata())

            # Loop over lines (in case the hue was set)
            for line_index in range(0, len(lines), x_len + 1):
                lineXs, lineYs = lines[line_index].get_data()

                # Loop over points in line
                for _x, _y in zip(lineXs, lineYs):
                    _fax.text(
                        x=_x,
                        y=_y,
                        s=unit_fmt.format(_y),
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=ML_PLOT_COLORS[color_theme]["text"],
                        fontsize=ML_PLOT_FONTS["text"],
                    )

        _fax.spines["bottom"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["top"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["right"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["left"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.xaxis.label.set_color(ML_PLOT_COLORS[color_theme]["xlabel"])
        _fax.yaxis.label.set_color(ML_PLOT_COLORS[color_theme]["ylabel"])

        if axis_limits is not None:
            _fax.set_xlim(axis_limits[:2])
            _fax.set_ylim(axis_limits[2:])

        if _axii == 0:
            _fax.set_ylabel(ylabel, fontsize=ML_PLOT_FONTS["ylabel"])
        _fax.set_xlabel(xlabel, fontsize=ML_PLOT_FONTS["xlabel"])

        # Set background color
        _fax.set_facecolor(ML_PLOT_COLORS[color_theme]["background"])

    sns.despine(left=True)

    # Set background color
    f.fig.patch.set_facecolor(ML_PLOT_COLORS[color_theme]["background"])

    return f


def distplot(
    data,
    x,
    hue=None,
    title=None,
    rotate_x_ticks=False,
    color_theme=ML_PLOT_GLOBAL_THEME,
    xlabel="",
    ylabel="",
    axis_limits=None,
    xticks_off=False,
    yticks_off=False,
    legend=False,
    legend_location="upper right",
    dist_args={},
    **kwargs
):

    f = sns.FacetGrid(data=data, hue=hue, **kwargs)
    f.map(sns.distplot, x, **dist_args)

    if rotate_x_ticks:
        plt.xticks(rotation=45, horizontalalignment="right")

    if title is not None:
        plt.title(
            title,
            fontsize=ML_PLOT_FONTS["title"],
            color=ML_PLOT_COLORS[color_theme]["title"],
        )

    if legend:

        leg = f.fig.legend(loc=legend_location, framealpha=0)

        for text in leg.get_texts():
            plt.setp(text, color=ML_PLOT_COLORS[color_theme]["legend"])

    if xticks_off:
        plt.xticks([])
    plt.xticks(
        color=ML_PLOT_COLORS[color_theme]["xticks"], fontsize=ML_PLOT_FONTS["xticks"]
    )

    if yticks_off:
        plt.yticks([])
    plt.yticks(
        color=ML_PLOT_COLORS[color_theme]["yticks"], fontsize=ML_PLOT_FONTS["yticks"]
    )

    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=True,
    )  # labels along the bottom edge are off

    for _fax in f.axes.flatten():

        _fax.spines["bottom"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["top"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["right"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.spines["left"].set_color(ML_PLOT_COLORS[color_theme]["spines"])
        _fax.xaxis.label.set_color(ML_PLOT_COLORS[color_theme]["xlabel"])
        _fax.yaxis.label.set_color(ML_PLOT_COLORS[color_theme]["ylabel"])

        if axis_limits is not None:
            _fax.set_xlim(axis_limits[:2])
            _fax.set_ylim(axis_limits[2:])

        _fax.set_ylabel(
            ylabel,
            fontsize=ML_PLOT_FONTS["ylabel"],
            color=ML_PLOT_COLORS[color_theme]["xlabel"],
        )
        _fax.set_xlabel(
            xlabel,
            fontsize=ML_PLOT_FONTS["xlabel"],
            color=ML_PLOT_COLORS[color_theme]["ylabel"],
        )

        # Set background color
        _fax.set_facecolor(ML_PLOT_COLORS[color_theme]["background"])

    sns.despine(left=True)

    # Set background color
    f.fig.patch.set_facecolor(ML_PLOT_COLORS[color_theme]["background"])

    return f


def mplot_reg_results(
    predictions,
    true_values,
    feature_list=None,
    lm_coefs=None,
    title=None,
    visibility="visible",
    n_important_features=20,
    data_transform=None,
    return_handle=False,
):
    """
    Generates an overview plot to visualize the performance of a (linear) regression model.

    Parameters
    ----------
    predictions: pandas.Series, numpy.array or list
        Predicted values from a regression model
    true_values: pandas.Series, numpy.array or list
        True labels of data
    feature_list: pandas.Series, numpy.array or list, optional
        Ordered list of feature names that are used by the model. When this is supplied a feature importance plot will be added.
    lm_coefs: pandas.Series, numpy.array or list, optional
        Ordered list of model coefficients, in the same order as `feature names`. When this is supplied a feature importance plot will be added.
    title: string, optional
        Title of the figure
    visibility: string, optional, deprecated
        DEPRECATED! Visibility setting of the underlying PyPlot figure.
    n_important_features: int, optional
        Number of features to include in feature importance plot. Default = 20
    data_transform: float, optional
        Parameter to project transformed data back into original space. Currently only supports lambda-parameter of BoxCox transform.
    return_handle: boolean, optional
        When set to True the figure handle will be returned in addition to displaying the figure. Default to False.

    Returns
    -------
    fig : matplotlib.pyplot figure, optional
        Can optionally return the figure handle
    """

    # Font properties
    sup_title_weight = "bold"
    sup_title_size = 24

    sub_title_weight = "bold"
    sub_title_size = 18

    axis_label_weight = "normal"
    axis_label_size = 14

    plot_df = pd.DataFrame(
        {"Predicted value": predictions, "Observed value": true_values}
    )

    # Kick out NaNs
    plot_df = plot_df.dropna()

    # Transform data if needed
    if data_transform is not None:
        plot_df["Observed value"] = inv_boxcox(
            plot_df["Observed value"].values, data_transform
        )
        plot_df["Predicted value"] = inv_boxcox(
            plot_df["Predicted value"].values, data_transform
        )

    # Compute performance measures
    # adjusted R2
    r2 = r2_score(y_pred=plot_df["Predicted value"], y_true=plot_df["Observed value"])
    if feature_list is not None:
        num_obs = plot_df.shape[0]
        num_feat = len(feature_list)
        r2 = 1 - (1 - r2) * ((num_obs - 1) / (num_obs - num_feat - 1))

    # MSLE
    # msle = mean_squared_log_error(y_pred=plot_df['Predicted value'], y_true=plot_df['Observed value'])

    # MAE
    mae = mean_absolute_error(
        y_pred=plot_df["Predicted value"], y_true=plot_df["Observed value"]
    )

    # Compute absolute and relative errors
    errors = plot_df["Observed value"] - plot_df["Predicted value"]

    # if data_transform is not None:
    #    errors     = inv_boxcox(true_values - predictions, data_transform)
    # else:
    #    errors     = plot_df['Observed value']-plot_df['Predicted value']

    abs_errors = np.abs(errors)
    rel_errors = 100 * (errors / plot_df["Observed value"])

    abs_errors = abs_errors[np.abs(rel_errors) <= 100]
    rel_errors = rel_errors[np.abs(rel_errors) <= 100]

    binned_abs_errors = np.percentile(abs_errors, np.arange(10, 110, 10))
    binned_rel_errors = np.percentile(np.abs(rel_errors), np.arange(10, 110, 10))

    # Build plot
    fig = plt.figure(figsize=(16, 16))

    if title is not None:
        plt.suptitle(title, fontweight=sup_title_weight, fontsize=sup_title_size)

    # Regression plot
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot(221)
        ax.set_title(
            "Regression model results",
            fontweight=sub_title_weight,
            fontsize=sub_title_size,
        )

        sns.regplot(
            ax=ax,
            y=plot_df["Predicted value"],
            x=plot_df["Observed value"],
            truncate=True,
            line_kws={"color": [1, 0, 0, 1]},
        )

        ax.set_xlabel(
            "Observed value", fontweight=axis_label_weight, fontsize=axis_label_size
        )
        ax.set_ylabel(
            "Predicted value", fontweight=axis_label_weight, fontsize=axis_label_size
        )

        # Add stats
        ax.text(
            0.9,
            0.2,
            s="$\overline{R}^2$:" + "{:0.2f}\nMAE: {:.1f}".format(r2, mae),
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor=(0.3, 0.3, 0.3), alpha=0.8),
        )

    # Error distribution
    with sns.axes_style("white"):
        ax1 = fig.add_subplot(422)
        ax1.set_title(
            "Error distribution", fontweight=sub_title_weight, fontsize=sub_title_size
        )
        barfig = sns.barplot(x=np.arange(10, 110, 10), y=binned_abs_errors, ax=ax1)
        # Add values on top of bars
        for _x, _y in enumerate(binned_abs_errors):
            barfig.text(_x, _y, "{:.0f}".format(_y), color="black", ha="center")
        ax1.set_ylabel(
            "Absolute error", fontweight=axis_label_weight, fontsize=axis_label_size
        )

        ax = fig.add_subplot(424, sharex=ax1)

        barfig2 = sns.barplot(x=np.arange(10, 110, 10), y=binned_rel_errors, ax=ax)
        # Add values on top of bars
        for _x, _y in enumerate(binned_rel_errors):
            barfig2.text(_x, _y, "{:.0f}%".format(_y), color="black", ha="center")
        ax.set_ylabel("% error", fontweight=axis_label_weight, fontsize=axis_label_size)

    # Distribution plot
    with sns.axes_style("white"):
        ax = fig.add_subplot(223)
        ax.set_title(
            "Value distribution", fontweight=sub_title_weight, fontsize=sub_title_size
        )
        long_df = pd.DataFrame(
            {
                "value": pd.concat(
                    [plot_df["Observed value"], plot_df["Predicted value"]],
                    axis=0,
                    ignore_index=True,
                ),
                "type": ["Observed"] * plot_df.shape[0]
                + ["Predicted"] * plot_df.shape[0],
            }
        )

        sns.distplot(
            long_df[long_df["type"] == "Observed"]["value"],
            hist=False,
            kde_kws={"shade": True, "legend": True},
            label="Observed",
            ax=ax,
        )
        sns.distplot(
            long_df[long_df["type"] == "Predicted"]["value"],
            hist=False,
            kde_kws={"shade": True, "legend": True},
            label="Predicted",
            ax=ax,
        )

        ax.set_xlabel("Value", fontweight=axis_label_weight, fontsize=axis_label_size)
        ax.set_ylabel("Density", fontweight=axis_label_weight, fontsize=axis_label_size)

    # Feature importance
    if (
        (feature_list is not None)
        & (lm_coefs is not None)
        & (len(feature_list) == len(lm_coefs))
    ):
        with sns.axes_style("white"):
            ax = fig.add_subplot(224)
            ax.set_title(
                "Feature weights", fontweight=sub_title_weight, fontsize=sub_title_size
            )
            feature_importance = pd.DataFrame(
                {"feature": feature_list, "weight": np.abs(lm_coefs)}
            ).sort_values("weight", ascending=False)

            sns.barplot(
                ax=ax,
                data=feature_importance[:n_important_features],
                x="weight",
                y="feature",
            )
            #     plt.stem(x=feature_importance[:n_important_features]['weight'].values, y=feature_importance[:n_important_features]['feature'].values)
            ax.yaxis.tick_right()
            ax.invert_xaxis()
            ax.set_xlabel(
                "Weight", fontweight=axis_label_weight, fontsize=axis_label_size
            )
            ax.set_ylabel("", fontweight=axis_label_weight, fontsize=axis_label_size)
    elif len(feature_list) != len(lm_coefs):
        print("ERROR: Length of feature list does not match length of coefficients!")

    sns.despine()
    sns.despine(ax=ax, left=True, right=False)

    #     fig.tight_layout()

    plt.subplots_adjust(left=0.1, wspace=0.2, top=0.9)

    if return_handle:
        return fig


def mplot_class_results(
    predictions,
    predictions_probabilities,
    true_values,
    feature_list=None,
    lr_coefs=None,
    class_labels=None,
    title=None,
    visibility="visible",
    n_important_features=20,
    return_handle=False,
):

    """
    Generates an overview plot to visualize the performance of a binary classification model.

    Parameters
    ----------
    predictions: pandas.Series, numpy.array or list
        Predicted values from a binary classification model
    prediction_probabilities: pandas.Series, numpy.array or list
        Probabilities (e.g. from a predict_proba function) associated with each prediction.
    true_values: pandas.Series, numpy.array or list
        True labels of data
    feature_list: pandas.Series, numpy.array or list, optional
        Ordered list of feature names that are used by the model. When this is supplied a feature importance plot will be added.
    lm_coefs: pandas.Series, numpy.array or list, optional
        Ordered list of model coefficients, in the same order as `feature names`. When this is supplied a feature importance plot will be added.
    class_labels: list, optional
        Alternative names for classes to use in figure. Should be in ascending order of class indices.
    title: string, optional
        Title of the figure
    visibility: string, optional, deprecated
        DEPRECATED! Visibility setting of the underlying PyPlot figure.
    n_important_features: int, optional
        Number of features to include in feature importance plot. Default = 20
    return_handle: boolean, optional
        When set to True the figure handle will be returned in addition to displaying the figure. Default to False.

    Returns
    -------
    fig : matplotlib.pyplot figure, optional
        Can optionally return the figure handle
    """

    # Font properties
    sup_title_weight = "bold"
    sup_title_size = 20

    sub_title_weight = "semibold"
    sub_title_size = 16

    axis_label_weight = "heavy"
    axis_label_size = 11

    # Put preds and actuals in df
    plot_df = pd.DataFrame(
        {"Predicted value": predictions, "Observed value": true_values}
    )

    # Kick out NaNs
    plot_df = plot_df.dropna()

    # Compute performance measures
    y_pred = predictions
    y_test = true_values

    # ROC
    y_pred_prob = predictions_probabilities
    y_pred_prob_class0 = y_pred_prob[:, 0]
    y_pred_prob_class1 = y_pred_prob[:, 1]  # save probability for class 1
    fpr, tpr, thresholds = roc_curve(
        y_test, y_pred_prob_class1, pos_label=true_values.unique()[1]
    )
    auc = roc_auc_score(y_test, y_pred_prob_class1)
    acc = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") * 100 / cm.sum(axis=1)[:, np.newaxis]

    # Build Plot
    fig = plt.figure(figsize=(14, 10))

    plt.suptitle(
        "Model accuracy: {:.1f}%".format(100 * acc),
        verticalalignment="top",
        fontweight=sup_title_weight,
        fontsize=sup_title_size,
        y=1.03,
    )

    # ROC-curve plot
    ax1 = plt.subplot2grid((2, 9), (0, 0), colspan=3)
    sns.lineplot(x=fpr, y=tpr, ax=ax1, ci=None)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("ROC curve", fontweight=sub_title_weight, fontsize=sub_title_size, pad=10)
    plt.xlabel(
        "False Positive Rate (1 - Specificity)",
        fontweight=axis_label_weight,
        fontsize=axis_label_size,
    )
    plt.ylabel(
        "True Positive Rate (Sensitivity)",
        fontweight=axis_label_weight,
        fontsize=axis_label_size,
    )
    plt.grid(True, linestyle="--")

    ax1.text(
        0.8,
        0.2,
        s="AUC: {:.2f}".format(auc),
        transform=ax1.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", edgecolor=(0.3, 0.3, 0.3), alpha=0.8, pad=8),
    )

    # Confusion plot
    ax2 = plt.subplot2grid((2, 9), (0, 5), colspan=3)

    # build annotation for confusion matrix
    format_nrs = "{:,.0f}\n{:.0f}%"
    label = np.empty((cm.shape[0], 0)).tolist()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        label[i].append(format_nrs.format(cm[i, j], cm_norm[i, j]))
    label = np.array(label)

    sns.heatmap(
        cm_norm,
        annot=label,
        ax=ax2,
        fmt="",
        cmap="Blues",
        annot_kws={"size": 12, "va": "center", "ha": "center"},
        cbar_kws={"format": "%.0f%%"},
        vmin=0,
        vmax=100,
    )
    # annot=True to annotate cells
    # labels, title and ticks
    ax2.set_xlabel("Predicted", fontweight=axis_label_weight, fontsize=axis_label_size)
    ax2.set_ylabel("True", fontweight=axis_label_weight, fontsize=axis_label_size)
    ax2.set_title(
        "Confusion matrix", fontweight=sub_title_weight, fontsize=sub_title_size, pad=10
    )
    if class_labels != None:
        ax2.xaxis.set_ticklabels(class_labels, va="center")
        ax2.yaxis.set_ticklabels(class_labels, va="center")

    # Prediction probability histogram
    ax3 = plt.subplot2grid((3, 9), (2, 0), colspan=5)
    sns.distplot(y_pred_prob_class0, ax=ax3)
    sns.distplot(y_pred_prob_class1, ax=ax3)
    plt.title(
        "Predicted probabilities",
        fontweight=sub_title_weight,
        fontsize=sub_title_size,
        pad=10,
    )
    plt.xlabel(
        "Predicted probability", fontweight=axis_label_weight, fontsize=axis_label_size
    )
    plt.ylabel("Frequency", fontweight=axis_label_weight, fontsize=axis_label_size)
    plt.xlim(0, 1)
    plt.axvline(0.5, 0, 100, color="r", linewidth=1.5)
    plt.legend("Class 0", "Class 1")

    # # ax4 = plt.subplot2grid((2, 9), (1, 3), colspan=3)
    # sns.distplot(y_pred_prob_class1, ax=ax3)
    # plt.title('Predicted probabilities\nfor class 1', fontweight=sub_title_weight, fontsize=sub_title_size, pad=10);
    # plt.xlabel('Predicted probability', fontweight=axis_label_weight, fontsize=axis_label_size)
    # plt.ylabel('Frequency', fontweight=axis_label_weight, fontsize=axis_label_size)
    # plt.xlim(0,1)
    # plt.axvline(0.5, 0, 100, color='r', linewidth=1.5)

    # Feature importance plot
    if (
        (feature_list is not None)
        & (lr_coefs is not None)
        & (len(feature_list) == len(lr_coefs))
    ):
        ax5 = plt.subplot2grid((3, 9), (2, 5), colspan=3)
        with sns.axes_style("white"):
            ax5.set_title(
                "Feature weights",
                fontweight=sub_title_weight,
                fontsize=sub_title_size,
                pad=10,
            )
            feature_importance = pd.DataFrame(
                {"feature": feature_list, "weight": np.abs(lr_coefs)}
            ).sort_values("weight", ascending=False)

            sns.barplot(
                ax=ax5,
                data=feature_importance[:n_important_features],
                x="weight",
                y="feature",
            )
            #     plt.stem(x=feature_importance[:n_important_features]['weight'].values, y=feature_importance[:n_important_features]['feature'].values)
            ax5.yaxis.tick_right()
            ax5.invert_xaxis()
            ax5.set_xlabel(
                "Weight", fontweight=axis_label_weight, fontsize=axis_label_size
            )
            ax5.set_ylabel("", fontweight=axis_label_weight, fontsize=axis_label_size)
            ax5.yaxis.set_ticklabels(ax5.yaxis.get_ticklabels(), fontsize=9)

    plt.tight_layout()

    if return_handle:
        return fig


def plot_confusion_matrix(
    estimator, X_test, y_test, title="", cmap=plt.cm.Blues, suptitle=None
):
    """
    Wrapper function for invoking plot_one_cm(). Makes sure the correct parameters
    are used, and adds optinal title/suptitles.
    Result is a normalized and non-normalized plot of confusion matrix for classifier.

    Parameters
    ----------
    estimator : classification estimator
        Trained classification estimator
    X_test : pandas.DataFrame / pandas.Series
        Test set with input features for testing trained model
    y_test : pandas.Series
        Test set with target values for validating trained model
    title : string, optional
        Title to add above plot
    cmap : matplotlib colormap, optional
        Colormap to use for plotting confusion matrix
    suptitle : string
        Suptitle to add above title, optional
    """

    fig, ax = plt.subplots(1, 1)

    classes = sorted(y_test.unique())
    cnf_matrix = confusion_matrix(y_test, estimator.predict(X_test))

    plot_one_cm(cnf_matrix, classes, ax=ax)

    if suptitle != None:
        plt.suptitle(suptitle, size=21, va="top")
    plt.tight_layout()
    plt.show()


def plot_one_cm(cm, classes, title="", prop_to_perc=False, cmap=plt.cm.Blues, ax=None):
    """
    This function plots the confusion matrix for a classifier (can be multi-class).
    Result is a normalized and non-normalized plot of the confusion matrix.

    Parameters
    ----------
    cm : sklearn.metrics.confusion_matrix
        An instance of the confusion_matrix class
    classes : list
        List with (unique) target classes
    title : string, optional
        Title to print above plot
    prop_to_perc : boolean, optional
        Convert proportions to percentages before plotting
    cmap : matplotlib colormap, optional
        Colormap to use for plotting confusion matrix
    ax : matplotlib axis object, optional

    """

    if ax is not None:
        plt.sca(ax)

    norm_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(norm_cm, interpolation="nearest", cmap=cmap)
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, size=16)
    plt.yticks(tick_marks, classes, size=16)

    format_nrs = "{:,}\n{:.3f}"

    if prop_to_perc:
        norm_cm *= 100
        format_nrs = "{:,}\n{:.1f}%"

    thresh = cm.max() / 1.5
    norm_thresh = norm_cm.max().max() / 1.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format_nrs.format(cm[i, j], norm_cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            #  color="black", size=14)
            color="white" if norm_cm[i, j] > norm_thresh else "black",
            size=14,
        )

    plt.ylabel("True label", size=18)
    plt.xlabel("Predicted label", size=18)
    plt.title(title, size=18)


def plot_feature_importance(
    mdl,
    importance_type="gain",
    feature_names=None,
    num_feat=20,
    return_handle=False,
    show_figure=True,
):
    import seaborn as sns
    from matplotlib import pyplot as plt
    from ml_tools.utils import get_feature_importances

    _df = get_feature_importances(
        mdl, importance_type=importance_type, named_features=feature_names
    ).sort_values(importance_type, ascending=False)

    _df = _df[:num_feat]

    plt.figure(figsize=(14, 6))
    g = sns.barplot(
        x=importance_type, y="Feature", data=_df[_df[importance_type] > 0], orient="h"
    )
    labels = g.get_yticklabels()
    g.set_yticklabels(labels, fontdict=dict(fontsize=5))
    plt.tight_layout()

    if show_figure:
        display(g.figure)

    if return_handle:
        return g.figure
