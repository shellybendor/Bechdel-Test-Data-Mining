# imports
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from collections import defaultdict
import gender_guesser.detector as gen


def get_data():
    # Getting data from Bechdel Test Api
    df = pd.read_json('http://bechdeltest.com/api/v1/getAllMovies')
    # Removing movies from before 1967 (no movie passed before this year)
    df_recent = df[df['year']>=1967]
    # Data cleaning
    df_recent.rename(columns={'rating': 'Bechdel Score'}, inplace=True)
    df_recent['Bechdel Score'] = df_recent['Bechdel Score'].astype('category',
                                                           copy=False)
    passed_test_column = []
    for i in df_recent['Bechdel Score']:
        if i < 3:
            passed_test_column.append(0)
        else:
            passed_test_column.append(1)
    df_recent['Passed Test'] = passed_test_column
    return df_recent


def _plot_test_scores_over_time(df):
    counter_years = Counter(df["year"])
    counter_score_3 = Counter(df.loc[df["Bechdel Score"] == 3, "year"])
    counter_score_0 = Counter(df.loc[df["Bechdel Score"] == 0, "year"])
    counter_score_1 = Counter(df.loc[df["Bechdel Score"] == 1, "year"])
    counter_score_2 = Counter(df.loc[df["Bechdel Score"] == 2, "year"])
    normalized__score_3 = {year: count / counter_years[year] for year, count in
                           dict(counter_score_3).items()}
    normalized__score_2 = {year: count / counter_years[year] for year, count in
                           dict(counter_score_2).items()}
    normalized__score_1 = {year: count / counter_years[year] for year, count in
                           dict(counter_score_1).items()}
    normalized__score_0 = {year: count / counter_years[year] for year, count in
                           dict(counter_score_0).items()}
    plt.plot(dict(normalized__score_3).keys(),
             dict(normalized__score_3).values(), label="Movies with score 3",
             marker='.')
    plt.plot(dict(normalized__score_2).keys(),
             dict(normalized__score_2).values(), label="Movies with score 2",
             marker='.')
    plt.plot(dict(normalized__score_1).keys(),
             dict(normalized__score_1).values(), label="Movies with score 1",
             marker='.')
    plt.plot(dict(normalized__score_0).keys(),
             dict(normalized__score_0).values(), label="Movies with score 0",
             marker='.')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Bechdel Test Scores in Movies over Time')
    plt.legend()
    plt.show()


def _plot_passed_vs_failed_over_time(df):
    counter_years = Counter(df["year"])
    counter_passed = Counter(df.loc[df["Passed Test"] == 1, "year"])
    counter_failed = Counter(df.loc[df["Passed Test"] == 0, "year"])
    normalized_passed = {year: count / counter_years[year] for year, count in
                           dict(counter_passed).items()}
    normalized_failed = {year: count / counter_years[year] for year, count in
                           dict(counter_failed).items()}
    plt.plot(dict(normalized_passed).keys(),
             dict(normalized_passed).values(), label="Movies that passed",
             marker='.')
    plt.plot(dict(normalized_failed).keys(),
             dict(normalized_failed).values(), label="Movies that failed",
             marker='.')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Movies passed and failed over time')
    plt.legend()
    plt.show()


def get_df_from_tmdb():
    df = pd.read_json("bechdel_movies.json")
    return df


def _plot_imdb_rating_and_test_scores(imdb_df, bechdel_df):
    imdb_filtered = imdb_df[['title', 'vote_average']]
    merged_df = pd.merge(bechdel_df, imdb_filtered, how="left", left_on=['title'], right_on=["title"])
    merged_df = merged_df.dropna()
    # Creating a new dataframe with only year, Bechdel scores, and imdb rating:
    new = merged_df.groupby(['year', 'Passed Test']).agg(
        {'vote_average': 'mean'}).reset_index()

    # Plotting vote average for passing movies
    x_passed = new.loc[new["Passed Test"] == 1]['year']
    y_passed = new.loc[new["Passed Test"] == 1]['vote_average']
    plt.plot(x_passed, y_passed, 'o', color='darkblue')
    m, b = np.polyfit(x_passed, y_passed, 1)
    plt.plot(x_passed, m*x_passed+b, label="Passing movies", color='darkblue')

    # Plotting vote average for failing movies
    x_failed = new.loc[new["Passed Test"] == 0]['year']
    y_failed = new.loc[new["Passed Test"] == 0]['vote_average']
    plt.plot(x_failed, y_failed, 'o', color='orange')
    m, b = np.polyfit(x_failed, y_failed, 1)
    plt.plot(x_failed, m * x_failed + b, label="Failing movies",
             color='orange')

    plt.xlabel('Year')
    plt.ylabel('IMDB rating')
    plt.title('Imdb ratings avg in passing vs failing movies')
    plt.legend()
    plt.show()


def _plot_budget_and_test_scores(imdb_df, bechdel_df):
    imdb_filtered = imdb_df[['title', 'budget']]
    merged_df = pd.merge(bechdel_df, imdb_filtered, how="left", left_on=['title'], right_on=["title"])
    merged_df = merged_df.dropna()
    merged_df.drop(merged_df[merged_df["budget"] == 0].index, inplace=True)
    # Creating a new dataframe with only year, Bechdel scores, and imdb rating:
    new = merged_df.groupby(['year', 'Passed Test']).agg(
        {'budget': 'mean'}).reset_index()

    # Plotting vote average for passing movies
    x_passed = new.loc[new["Passed Test"] == 1]['year']
    y_passed = new.loc[new["Passed Test"] == 1]['budget']
    plt.plot(x_passed, y_passed, 'o', color='darkblue')
    m, b = np.polyfit(x_passed, y_passed, 1)
    plt.plot(x_passed, m*x_passed+b, label="Budget of passing movies", color='darkblue')

    # Plotting vote average for failing movies
    x_failed = new.loc[new["Passed Test"] == 0]['year']
    y_failed = new.loc[new["Passed Test"] == 0]['budget']
    plt.plot(x_failed, y_failed, 'o', color='orange')
    m, b = np.polyfit(x_failed, y_failed, 1)
    plt.plot(x_failed, m * x_failed + b, label="Budget of failing movies",
             color='orange')

    # plt.yscale("log")
    plt.xlabel('Year')
    plt.ylabel('Movie budget')
    plt.title('Movie budget avg in passing vs failing movies')
    plt.legend()
    plt.show()


def _plot_revenue_and_test_scores(imdb_df, bechdel_df):
    imdb_filtered = imdb_df[['title', 'revenue']]
    merged_df = pd.merge(bechdel_df, imdb_filtered, how="left", left_on=['title'], right_on=["title"])
    merged_df = merged_df.dropna()
    merged_df.drop(merged_df[merged_df["revenue"] == 0].index, inplace=True)
    # Creating a new dataframe with only year, Bechdel scores, and imdb rating:
    new = merged_df.groupby(['year', 'Passed Test']).agg(
        {'revenue': 'mean'}).reset_index()

    # Plotting vote average for passing movies
    x_passed = new.loc[new["Passed Test"] == 1]['year']
    y_passed = new.loc[new["Passed Test"] == 1]['revenue']
    plt.plot(x_passed, y_passed, 'o', color='darkblue')
    m, b = np.polyfit(x_passed, y_passed, 1)
    plt.plot(x_passed, m*x_passed+b, label="Revenue of passing movies", color='darkblue')

    # Plotting vote average for failing movies
    x_failed = new.loc[new["Passed Test"] == 0]['year']
    y_failed = new.loc[new["Passed Test"] == 0]['revenue']
    plt.plot(x_failed, y_failed, 'o', color='orange')
    m, b = np.polyfit(x_failed, y_failed, 1)
    plt.plot(x_failed, m * x_failed + b, label="Revenue of failing movies",
             color='orange')

    plt.xlabel('Year')
    plt.ylabel('Movie Revenue')
    plt.title('Movie revenue avg in passing vs failing movies')
    plt.legend()
    plt.show()


def _get_passed_movies(df):
    df.drop(df[df["Passed Test"] == 0].index,
            inplace=True)
    return df


def _get_failed_movies(df):
    df.drop(df[df["Passed Test"] == 1].index,
            inplace=True)
    return df


def _pie_chart_of_genres(imdb_df, bechdel_df):
    imdb_filtered = imdb_df[['title', 'genres']]
    merged_df = pd.merge(_get_passed_movies(bechdel_df), imdb_filtered, how="left",
                         left_on=['title'], right_on=["title"])
    merged_df = merged_df.dropna()
    count_genres = defaultdict(int)
    num_genres = 0
    num_passed_movies = merged_df.shape[0]
    for movie in merged_df["genres"]:
        for genre in movie:
            count_genres[genre["name"]] += 1
            num_genres += 1

    y = np.array(list(count_genres.values()))
    x = np.char.array(list(count_genres.keys()))
    colors = ['yellowgreen', 'red', 'gold', 'lightskyblue', 'teal',
              'lightcoral', 'blue', 'pink', 'darkgreen', 'yellow', 'grey',
              'violet', 'magenta', 'cyan', 'lightgreen', 'orange', 'purple',
              "brown", 'darkblue']
    porcent = 100. * y / y.sum()
    patches, texts = plt.pie(y, startangle=90, radius=1.2, colors=colors)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x , porcent)]
    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                             key=lambda x: x[2],
                                             reverse=True))
    plt.legend(patches, labels, bbox_to_anchor=(-0.1, 1.),
               fontsize=8)
    plt.show()


def _plot_director_gender(bechdel_df):
    latest = pd.read_csv('movielatest.csv', encoding="latin")
    dfLatest = latest[['name', 'director']]
    dfLatest.rename(columns={'name': 'title'}, inplace=True)
    dfLatest = pd.merge(bechdel_df, dfLatest, how='left', left_on=['title'],
                        right_on=['title'])
    dfLatest = dfLatest.dropna()

    # Predicting gender of director from first name:
    d = gen.Detector()
    genders = []
    firstNames = dfLatest['director'].str.split().str.get(0)
    for name in firstNames:
        if d.get_gender(name) == 'male':
            genders.append('male')
        elif d.get_gender(name) == 'female':
            genders.append('female')
        else:
            genders.append('unknown')
    dfLatest['gender'] = genders
    dfLatest = dfLatest[dfLatest['gender'] != 'unknown']
    # Encode the variable gender into a new dataframe:
    dfLatest['Male'] = dfLatest['gender'].map({'male': 1, 'female': 0})

    df_female = dfLatest[dfLatest['gender'] == 'female']["Passed Test"].value_counts(normalize=True)
    df_male = dfLatest[dfLatest['gender'] == 'male']["Passed Test"].value_counts(normalize=True)
    # print(df_female, df_male)
    labels = ["Failed Test", "Passed Test"]
    men_means = [df_male[0], df_male[1]]
    women_means = [df_female[0], df_female[1]]
    # print(men_means, women_means)
    plt.bar(np.arange(2), men_means, 0.35, label="Men")
    plt.bar(np.arange(2) + 0.35, women_means, 0.35, label="Women")
    plt.title('Genders of directors of passed and failed movies')
    plt.xticks(np.arange(2) + 0.35 / 2, labels)
    plt.ylabel("Percentage of total directors of each gender")
    plt.legend()
    for index, data in enumerate(men_means):
        plt.text(x=index, y=data, s=f"{round(data, 2)}")
    for index, data in enumerate(women_means):
        plt.text(x=index + 0.25, y=data, s=f"{round(data, 2)}")
    plt.tight_layout()
    plt.show()


def _plot_director_gender_by_year(bechdel_df):
    latest = pd.read_csv('movielatest.csv', encoding="latin")
    dfLatest = latest[['name', 'director']]
    dfLatest.rename(columns={'name': 'title'}, inplace=True)
    dfLatest = pd.merge(bechdel_df, dfLatest, how='left', left_on=['title'],
                        right_on=['title'])
    dfLatest = dfLatest.dropna()
    dfLatest = dfLatest[dfLatest['year'] >= 1986]

    # Predicting gender of director from first name:
    d = gen.Detector()
    genders = []
    firstNames = dfLatest['director'].str.split().str.get(0)
    for name in firstNames:
        if d.get_gender(name) == 'male':
            genders.append('male')
        elif d.get_gender(name) == 'female':
            genders.append('female')
        else:
            genders.append('unknown')
    dfLatest['gender'] = genders
    dfLatest = dfLatest[dfLatest['gender'] != 'unknown']
    # Encode the variable gender into a new dataframe:
    dfLatest['Male'] = dfLatest['gender'].map({'male': 1, 'female': 0})

    total_female_counts = dict(dfLatest[dfLatest["gender"] == 'female'][
            'year'].value_counts())
    female_counts = \
        _get_passed_movies(dfLatest[dfLatest["gender"] == 'female'])[
            'year'].value_counts().rename_axis("year").reset_index(name="counts")
    female_counts = female_counts.apply(lambda x:
                                        [x["year"], x['counts'] /
                                         total_female_counts[x["year"]]],
                                        axis=1, result_type='expand')


    total_male_counts = dict(dfLatest[dfLatest["gender"] == 'male'][
        'year'].value_counts())
    male_counts = \
    _get_passed_movies(dfLatest[dfLatest["gender"] == 'male'])[
        'year'].value_counts().rename_axis("year").reset_index(name="counts")
    male_counts = male_counts.apply(lambda x:
                                        [x["year"], x['counts'] /
                                         total_male_counts[x["year"]]],
                                        axis=1, result_type='expand')
    x_female = female_counts[0]
    y_female = female_counts[1]
    plt.plot(x_female, y_female, 'o', color='orange')
    m, b = np.polyfit(x_female, y_female, 1)
    plt.plot(x_female, m * x_female + b, label="female directors",
             color='orange')
    x_male = male_counts[0]
    y_male = male_counts[1]
    plt.plot(x_male, y_male, 'o', color='darkblue')
    m, b = np.polyfit(x_male, y_male, 1)
    plt.plot(x_male, m * x_male + b, label="male directors",
             color='darkblue')
    plt.xlabel('Year')
    plt.ylabel('Percentage of directors of same gender')
    plt.title('Percentage of directors of passing movies')
    plt.legend()
    plt.show()


def _plot_writer_gender_by_year(bechdel_df):
    latest = pd.read_csv('movielatest.csv', encoding="latin")
    dfLatest = latest[['name', 'writer']]
    dfLatest.rename(columns={'name': 'title'}, inplace=True)
    dfLatest = pd.merge(bechdel_df, dfLatest, how='left', left_on=['title'],
                        right_on=['title'])
    dfLatest = dfLatest.dropna()
    dfLatest = dfLatest[dfLatest['year'] >= 1986]

    # Predicting gender of writer from first name:
    d = gen.Detector()
    genders = []
    firstNames = dfLatest['writer'].str.split().str.get(0)
    for name in firstNames:
        if d.get_gender(name) == 'male':
            genders.append('male')
        elif d.get_gender(name) == 'female':
            genders.append('female')
        else:
            genders.append('unknown')
    dfLatest['gender'] = genders
    dfLatest = dfLatest[dfLatest['gender'] != 'unknown']
    # Encode the variable gender into a new dataframe:
    dfLatest['Male'] = dfLatest['gender'].map({'male': 1, 'female': 0})

    total_female_counts = dict(dfLatest[dfLatest["gender"] == 'female'][
            'year'].value_counts())
    female_counts = \
        _get_passed_movies(dfLatest[dfLatest["gender"] == 'female'])[
            'year'].value_counts().rename_axis("year").reset_index(name="counts")
    female_counts = female_counts.apply(lambda x:
                                        [x["year"], x['counts'] /
                                         total_female_counts[x["year"]]],
                                        axis=1, result_type='expand')


    total_male_counts = dict(dfLatest[dfLatest["gender"] == 'male'][
        'year'].value_counts())
    male_counts = \
    _get_passed_movies(dfLatest[dfLatest["gender"] == 'male'])[
        'year'].value_counts().rename_axis("year").reset_index(name="counts")
    male_counts = male_counts.apply(lambda x:
                                        [x["year"], x['counts'] /
                                         total_male_counts[x["year"]]],
                                        axis=1, result_type='expand')
    x_female = female_counts[0]
    y_female = female_counts[1]
    plt.plot(x_female, y_female, 'o', color='orange')
    m, b = np.polyfit(x_female, y_female, 1)
    plt.plot(x_female, m * x_female + b, label="female writers",
             color='orange')
    x_male = male_counts[0]
    y_male = male_counts[1]
    plt.plot(x_male, y_male, 'o', color='darkblue')
    m, b = np.polyfit(x_male, y_male, 1)
    plt.plot(x_male, m * x_male + b, label="male writers",
             color='darkblue')
    plt.xlabel('Year')
    plt.ylabel('Percentage of writers of same gender')
    plt.title('Percentage of writers of passing movies')
    plt.legend()
    plt.show()


def _get_words_for_word_cloud(bechdel_df):
    latest = pd.read_csv('movielatest.csv', encoding="latin")
    dfLatest = latest[['name', 'director', 'star', 'writer']]
    dfLatest.rename(columns={'name': 'title'}, inplace=True)
    dfLatest = pd.merge(bechdel_df, dfLatest, how='left', left_on=['title'],
                        right_on=['title'])
    dfLatest = _get_passed_movies(dfLatest.dropna())

    words = []
    for comp in dfLatest['director']:
        words.append(comp.replace(" ", "-"))
    for comp in dfLatest['star']:
        words.append(comp.replace(" ", "-"))
    for comp in dfLatest['writer']:
        words.append(comp.replace(" ", "-"))
    print(' '.join(words))


if __name__ == '__main__':
    # pre-processing data and sanity checks
    bechdel_df = get_data()
    tmdb_df = get_df_from_tmdb()

    # for each different graph we used *one* of the following functions and
    # commented out the rest
    # _plot_test_scores_over_time(bechdel_df)
    # _plot_passed_vs_failed_over_time(bechdel_df)
    # _plot_imdb_rating_and_test_scores(tmdb_df, bechdel_df)
    # _plot_budget_and_test_scores(tmdb_df, bechdel_df)
    # _plot_revenue_and_test_scores(tmdb_df, bechdel_df)
    # _pie_chart_of_genres(tmdb_df, bechdel_df)
    # _plot_director_gender(bechdel_df)
    _plot_director_gender_by_year(bechdel_df)
    # _plot_writer_gender_by_year(bechdel_df)
    # _get_words_for_word_cloud(bechdel_df)
