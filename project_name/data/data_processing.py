import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations


def open_File(name):
    """
    Opens a JSON  file and returns the data as a DataFrame.
    :param name: name of the JSON file
    :return: DataFrame with data
    """
    data = pd.read_json(name, lines=True)
    return data


def remove_columns(data):
    """
    Removes unnecessary columns from the DataFrame.
    :param data: original DataFrame
    :return: cleaned DataFrame
    """
    data = data.drop(columns=['Artist(s)', 'song', 'Length', 'emotion',
                              'Album', 'Release Date', 'Key', 'Tempo',
                              'Loudness (db)', 'Time signature', 'Explicit',
                              'Popularity', 'Energy', 'Danceability',
                              'Positiveness', 'Speechiness', 'Liveness',
                              'Acousticness', 'Instrumentalness',
                              'Good for Party', 'Good for Work/Study',
                              'Good for Relaxation/Meditation',
                              'Good for Exercise', 'Good for Running',
                              'Good for Yoga/Stretching', 'Good for Driving',
                              'Good for Social Gatherings',
                              'Good for Morning Routine', 'Similar Songs'])
    return data


def save_data(data, name):
    """
    Saves a DataFrame to a JSON file.
    :param data: DataFrame to be saved
    :param name: name of the output JSON file
    :return: None
    """
    data.to_json(data.to_json(name, orient='records',
                              lines=True))


def explode_genres(data):
    """
    Splits genre strings into lists of genres.
    :param data: DataFrame with 'Genre' column
    :return: modified DataFrame
    """
    # Split the genre string by comma, strip whitespace, drop empty values
    data['Genre'] = data['Genre'].astype(str).str.split(',')
    data['Genre'] = data['Genre'].apply(lambda genres: [g.strip() for g in genres if g.strip()])
    return data


def getGenreNames(data):
    """
    Prints the number and names of unique genres.
    :param data: DataFrame with 'Genre' column
    :return: None
    """
    genreSet = set()
    for genres in data['Genre']:
        genreSet.update(genres)
    print(len(genreSet))
    print(genreSet)



def plot_genre_counts(data):
    """
    Plots the amount of songs per genre
    :param data: data to be plotted
    :return: None
    """
    if 'Genre' not in data.columns:
        print("No 'Genre' column found.")
        return

    genre_series = data.copy()
    genre_series = genre_series.explode('Genre')['Genre']
    genre_counts = genre_series.value_counts()
    print(genre_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Number of Songs')
    plt.title('Number of Songs per Genre')
    plt.tight_layout()
    plt.show()


def plot_heatmap(data):
    """
    Plots a heatmap of genre co-occurrence.
    :param data: DataFrame with exploded 'Genre' lists
    :return: None
    """
    cooccurrence = Counter()

    for genres in data['Genre']:
        unique_genres = sorted(set(genres))
        if len(unique_genres) > 1:
            for g1, g2 in combinations(unique_genres, 2):
                cooccurrence[(g1, g2)] += 1
                cooccurrence[(g2, g1)] += 1  # symmetric

    genres = sorted({g for pair in cooccurrence for g in pair})
    matrix = pd.DataFrame(0, index=genres, columns=genres)

    for (g1, g2), count in cooccurrence.items():
        matrix.loc[g1, g2] = count

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.title("Rock Subgenre Co-Occurrence Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def filter_only_rock_genres(data):
    """
    Keep only songs with at least one rock-related genre,
    and remove "rock" (the general tag) from their genre list.
    :param data: DataFrame with genre information
    :return: filtered DataFrame
    """
    def filter_rock_subgenres(genre_list):
        return [g for g in genre_list if 'rock' in g.lower() and g.lower().strip() != 'rock']

    # Keep only rows with at least one specific rock subgenre
    data = data.copy()
    data['Genre'] = data['Genre'].apply(filter_rock_subgenres)

    data = data[data['Genre'].apply(len) > 0].reset_index(drop=True)

    return data


def openAndRefactor(name):
    """
    Opens and optinally refactors the data file. 
    :param data: name of the input file
    :return: loaded DataFrame
    """
    data_name = name
    data = open_File(data_name)

    if data_name == 'data.json':
        refactored_data = remove_columns(data)
        save_data(refactored_data, 'finished_data.json')
    return data


def processData(data):
    """
    Prepares the data for analysis.
    :param data: raw DataFrame
    :return: processed DataFrame
    """
    data = explode_genres(data)
    getGenreNames(data)
    data = filter_only_rock_genres(data)

    return data


def plot(data):
    """
    Plots genre counts and co-occurrences.
    :param data: processed DataFrame
    :return: None
    """
    plot_genre_counts(data)
    plot_heatmap(data)


def main():
    """
    Main function to run the full pipeline.
    :return: None
    """
    original_data = openAndRefactor('data.json')
    data = processData(original_data)
    # save_data(data, 'finished_data.json')
    plot(data)


if __name__ == "__main__":
    main()
