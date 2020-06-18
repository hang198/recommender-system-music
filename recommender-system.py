import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import turicreate as tc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.float_format', lambda x: '%.2f' % x)

user_data = pd.read_table('data/usersha1-artmbid-artname-plays.tsv',
                          header=None, nrows=2e7,
                          names=['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols=['users', 'artist-name', 'plays'])
user_profiles = pd.read_table('data/usersha1-profile.tsv',
                              header=None,
                              names=['users', 'gender', 'age', 'country', 'signup'],
                              usecols=['users', 'country'])
print(user_data.head())
print(user_profiles.head())
if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis=0, subset=['artist-name'])

artist_plays = (user_data.
    groupby(by=['artist-name'])['plays'].
    sum().
    reset_index().
    rename(columns={'plays': 'totalArtistPlays'})
[['artist-name', 'totalArtistPlays']])
# print(artist_plays.head())

user_data_with_artist_plays = user_data.merge(artist_plays,
                                              left_on='artist-name', right_on='artist-name', how='left')
user_data_with_artist_plays = user_data_with_artist_plays.rename(columns={"plays": "userArtistPlays"})
# print(user_data_with_artist_plays.head())
# print(artist_plays['totalArtistPlays'].describe())

# print(artist_plays['totalArtistPlays'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 40000
user_data_popular_artists = user_data_with_artist_plays.query('totalArtistPlays >= @popularity_threshold')
user_data_popular_artists.head()

combined = user_data_popular_artists.merge(user_profiles,
                                           left_on='users', right_on='users', how='left')
usa_data = combined.query('country == \'United States\'')
# print(usa_data.head())
if not usa_data[usa_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = usa_data.shape[0]

    # print('Initial dataframe shape {0}'.format(usa_data.shape))
    usa_data = usa_data.drop_duplicates(['users', 'artist-name'])
    current_rows = usa_data.shape[0]
    # print('New dataframe shape {0}'.format(usa_data.shape))
    # print('Removed {0} rows'.format(initial_rows - current_rows))


def data_to_sparse(data, index, columns, values):
    pivot = data.pivot(index=index, columns=columns, values=values).fillna(0)
    sparse = csr_matrix(pivot.values)
    print(sparse.shape)
    return pivot, sparse


def fit_knn(sparse):
    knn = NearestNeighbors(metric='cosine')
    knn.fit(sparse)
    print(knn)
    return knn


pivot_usa, sparse_usa = data_to_sparse(usa_data, index='artist-name', columns='users', values='userArtistPlays')
knn = fit_knn(sparse_usa)


# print(pivot_usa.head())

def idx_recommend(data, idx, model, k):
    distances, indices = (model.kneighbors(data.
                                           iloc[idx, :].
                                           values.reshape(1, -1),
                                           n_neighbors=k + 1))
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print(('Recommendations for artist {}:\n'.
                   format(data.index[idx])))
        else:
            print(('{}: {} ({:.3f})'.
                   format(i,
                          data.index[indices.flatten()[i]],
                          distances.flatten()[i])))
    return ''


# query_index = np.random.choice(pivot_usa.shape[0])
# idx_recommend(pivot_usa, query_index, knn, 6)

query_index = pivot_usa.index.get_loc('jan hammer')
idx_recommend(pivot_usa, query_index, knn, 6)

usa_data_zero_one = usa_data
usa_data_zero_one['everPlayed'] = usa_data_zero_one['userArtistPlays'].apply(np.sign)

pivot_usa_zero_one, sparse_usa_zero_one = data_to_sparse(usa_data_zero_one, index='artist-name', columns='users',
                                                         values='everPlayed')

knn = fit_knn(sparse_usa_zero_one)
idx_recommend(pivot_usa_zero_one, query_index, knn, 6)


def fuzzy_match(query_subject, data):
    ratio_tuples = []
    for subject in data.index:
        ratio = fuzz.ratio(subject.lower(), query_subject.lower())
        current_query_idx = data.index.tolist().index(subject)
        ratio_tuples.append((subject, ratio, current_query_idx))

    ratio_tuples = sorted(ratio_tuples, key=lambda tup: tup[1], reverse=True)[:2]
    print('Top matches: {}\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    match = ratio_tuples[0][0]
    return match, ratio_tuples


def print_artist_recommend(query_artist, data, model, k):
    match, ratio_tuples = fuzzy_match(query_artist, data)
    idx_recommend(data, ratio_tuples[0][2], model, k)
    return ''


# print_artist_recommend('jan hammer', pivot_usa_zero_one, knn, 10)

sf = tc.SFrame(usa_data_zero_one)
sf.head()

train, test = tc.recommender.util.random_split_by_user(sf,
                                                       user_id='users',
                                                       item_id='artist-name')


def train_models(train_data, test_data, user_id, item_id):
    # popularity
    m1 = tc.popularity_recommender.create(train_data,
                                          user_id=user_id,
                                          item_id=item_id,
                                          verbose=False
                                          )
    # similarity
    m2 = tc.item_similarity_recommender.create(train_data,
                                               user_id=user_id,
                                               item_id=item_id,
                                               verbose=False,
                                               similarity_type='jaccard'
                                               )

    model_list = m1, m2

    name_list = ['1. Popularity (Implicit)',
                 '2. Item Similarity (Implicit)'
                 ]

    results = tc.recommender.util.compare_models(test_data,
                                                 models=[m for m in model_list],
                                                 model_names=[n for n in name_list],
                                                 metric='auto')

    return model_list, name_list, results


model_list, name_list, results = train_models(train,
                                              test,
                                              user_id='users',
                                              item_id='artist-name',
                                              )
# determine a user list for which to pull custom recommendations
rand_users = np.random.choice(sf['users'], 5)
user_list = list(rand_users)


def played_vs_rec(users, df, model_list, name_list, user_id, item_id, plays):
    assert len(model_list) == len(name_list)
    print("User List: {}\n".format(users))
    df = df[[user_id, item_id, plays]]
    for user in users:
        print('USER {}\n'.format(str(user)))
        print(df[df[user_id] == user].sort(plays, ascending=False))
        print('\n')
        print('***RECOMMENDATIONS***')
        for i in range(len(model_list)):
            print('***{}***'.format(name_list[i]))
            results = model_list[i].recommend(users=[user], k=4)
            print(results)
        print('\n')


played_vs_rec(user_list,
              sf,
              model_list,
              name_list,
              user_id='users',
              item_id='artist-name',
              plays='userArtistPlays')

artist_list = [
    'Britney Spears',
    'Snoop Dogg',
    'Eminem',
    'Beatles',
    'D Bowie'
]


def find_similar(artist_list, model_list, data):
    for artist in artist_list:
        print(artist.upper())
        fuzz = process.extract(artist, data, limit=2)  # applying fuzzy matching
        print(model_list[1].get_similar_items([fuzz[0][0]], k=5))
    return ''


find_similar(artist_list, model_list, sf['artist-name'])
