import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def model_importances_plot(data):
    values = data.Importance    
    idx = data.Feature
    plt.figure(figsize=(12,6))
    clrs = ['green' if (x < max(values)) else 'red' for x in values ]
    sns.barplot(y=idx,x=values,palette=clrs).set(title='Important features')
    plt.show()

    return None

def distribution_target(df)
  fig = plt.figure(figsize=(8, 6))
  sns.histplot(df['target'], bins=10)
  plt.axvline(df['target'].median(), color='b', label='median')
  plt.axvline(df['target'].mean(), color='r', label='mean')
  plt.axvline(mode(sorted(round(df['target'],3)))[0][0], color='g', label='mode')
  plt.grid(linestyle='-.', color='grey')
  plt.legend();

def model_importances(estimator, data):
  feat_dict= {}
  for col, val in sorted(zip(data.columns, estimator.feature_importances_),key=lambda x:x[1],reverse=True):
    feat_dict[col]=val 
  return pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
     test and training learning curve.

    Parameters
    ----------
    estimator : object type estimator instance.

    X : array-like of shape (n_samples, n_features).

    y : array-like of shape (n_samples).

    cv : int, cross-validation generator or an iterable, default=None

    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=mape_score)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    print('train_scores_mean: ', train_scores_mean, '\ntest_scores_mean:', test_scores_mean)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt