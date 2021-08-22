# imports
import pandas as pd
import gender_guesser.detector as gen
import data_collector


def _encode_gender(dfLatest, position):
    # Predicting gender of writer from first name:
    d = gen.Detector()
    genders = []
    firstNames = dfLatest[position].str.split().str.get(0)
    for name in firstNames:
        if d.get_gender(name) == 'male':
            genders.append('male')
        elif d.get_gender(name) == 'female':
            genders.append('female')
        else:
            genders.append('unknown')
    dfLatest[position + ' gender'] = genders
    dfLatest = dfLatest[dfLatest[position + ' gender'] != 'unknown']
    # Encode the variable gender into a new dataframe:
    dfLatest['Male ' + position] = dfLatest[position + ' gender'].map({'male': 1, 'female': 0})
    dfLatest = dfLatest.drop([position, position + ' gender'], axis=1)
    return dfLatest


def seperate_categories(x, all_categories):
    cat = []
    for item in x:
        cat.append(item.get('name'))
        all_categories.add(item.get('name'))
    return cat


def _create_csv_from_classification(tmdb_df, bechdel_df):
    latest = pd.read_csv('movielatest.csv', encoding="latin")
    pd.set_option('display.max_columns', None)
    latest = latest[['name', 'company', 'country', 'year', 'director', 'writer', 'star']]
    latest.rename(columns={'name': 'title'}, inplace=True)
    merged_df = pd.merge(bechdel_df, tmdb_df, how="left",
                         left_on=['title'], right_on=["title"])
    merged_df = merged_df[['title', 'Passed Test', 'adult', 'genres', 'budget',
                           'popularity', 'revenue']]

    merged_df = merged_df[merged_df['budget'] > 0]
    merged_df = merged_df[merged_df['revenue'] > 0]
    merged_df = pd.merge(merged_df, latest, how='left', left_on=['title'],
                        right_on=['title'])
    merged_df = merged_df.dropna()
    merged_df = merged_df.drop_duplicates(subset=['title'])
    # creating gender features for predictions
    merged_df = _encode_gender(merged_df, 'director')
    merged_df = _encode_gender(merged_df, 'writer')
    merged_df = _encode_gender(merged_df, 'star')

    # creating genre features for predictions
    all_categories = set()
    merged_df['genres'] = merged_df['genres'].apply(lambda x: seperate_categories(x, all_categories))

    for item in list(all_categories):
        merged_df[item] = merged_df['genres'].apply(lambda x: 1 if item in x else 0)

    # create dummies for country and company
    merged_df = pd.get_dummies(merged_df, columns=["country", "company"])

    merged_df = merged_df.drop(['genres', 'title'], axis=1)
    merged_df.to_csv("bechdel_csv_for_predictions.csv")


if __name__ == '__main__':
    bechdel_df = data_collector.get_data()
    tmdb_df = data_collector.get_df_from_tmdb()
    # function for creating csv for prediction stage
    _create_csv_from_classification(tmdb_df, bechdel_df)