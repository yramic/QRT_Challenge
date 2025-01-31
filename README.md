# QRT Stock Prediction Challenge

This repository presents a possible solution to the [Stock Prediction Challenge](https://challengedata.ens.fr/participants/challenges/23/) by QRT. The primary focus of this challenge is to develop sophisticated feature engineering strategies. Given a stream of returns and traded volume rates for each specific identifier, domain knowledge plays a crucial role in crafting effective features.

## Methodology

### Feature Engineering

Feature engineering is the most critical aspect of this challenge. It became evident during this project that data is key and not so much the model and approach for the model chosen. By leveraging domain expertise, we can generate meaningful features that enhance model performance. The dataset provides return streams and trading volume rates, which can be transformed into predictive signals through various statistical and time-series techniques.

### Model Selection

Recently, at the [NeurlIPS](https://github.com/autogluon/neurips-autogluon-workshop) conference, Amazon Research introduced an improved version of [AutoGluon](https://auto.gluon.ai/stable/index.html), which is well-suited for tabular data with over 10,000 identifiers. AutoGluon has demonstrated strong performance in several Kaggle competitions, utilizing an ensemble of diverse models, ranging from decision trees to foundation models, with automatic hyperparameter tuning.

Unfortunately, running AutoGluon locally proved impractical due to high resource demands. Even executing it on a cluster did not yield significant improvements. As a result, I opted for well-established state-of-the-art models, such as [CatBoost](https://catboost.ai/), known for its efficiency and strong predictive power.

## Alternative Approaches

An alternative to AutoGluon is constructing a custom ensemble of models to frame stock return prediction as a binary classification task. Given the extensive feature set (around 100 engineered features), further improvements could be achieved through systematic hyperparameter tuning of AutoGluon. This approach may lead to superior performance compared to individual models like CatBoost.

## Conclusion

While AutoGluon offers a promising automated approach, its computational demands make it challenging to deploy effectively. Consequently, leveraging domain knowledge for feature engineering and using efficient models like CatBoost remains a practical and effective strategy for tackling the QRT Stock Prediction Challenge.