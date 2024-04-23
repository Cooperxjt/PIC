import os
import random

import pandas as pd

# This function splits annotation data from a CSV file into training and validation sets.
# The split is based on the number of images specified by the user.
#
# Parameters:
# csv_file_path: The path to the CSV file containing annotation data.
# N: The number of images to include in the training set.
#
# The function groups annotations by image, randomly selects N images for the training set,
# and uses the remaining images for the validation set.
# It then saves these annotations into two separate CSV files:
# one for training (username_train.csv) and one for validation (username_val.csv).


def split_annotations_for_folder(folder_path, N):
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(
        folder_path) if file.endswith('.csv')]

    for csv_file in csv_files:
        # Full path to the CSV file
        csv_file_path = os.path.join(folder_path, csv_file)

        # Load the CSV file
        df = pd.read_csv(csv_file_path)

        # Extract the username from the file name
        user_name = os.path.splitext(csv_file)[0]

        # Group data by image name
        grouped = df.groupby('name')

        # Select N images randomly
        selected_images = grouped.groups.keys()
        selected_images = list(selected_images)
        random.shuffle(selected_images)

        train_images = selected_images[:N]

        # Split the data into train and validation sets
        train_df = df[df['name'].isin(train_images)]
        val_df = df[~df['name'].isin(train_images)]

        if not os.path.isdir('test/' + str(N) + '/train/'):
            os.makedirs('test/' + str(N) + '/train')

        if not os.path.isdir('test/' + str(N) + '/val/'):
            os.makedirs('test/' + str(N) + '/val')

        # Save to new CSV files
        train_df.to_csv(os.path.join(
            'test/' + str(N) + '/train/', f'{user_name}_train.csv'), index=False)
        val_df.to_csv(os.path.join(
            'test/' + str(N) + '/val/', f'{user_name}_val.csv'), index=False)


# Example usage:
split_annotations_for_folder('test/total', N=20)
# Replace '/path/to/your/folder' with the actual folder path containing the CSV files and N with the desired number of images for training.
