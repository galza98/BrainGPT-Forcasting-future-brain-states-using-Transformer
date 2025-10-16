import os
import glob
import shutil
import pandas as pd
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

pd.set_option('display.max_rows', 3500)

class MoviefMRIDataset(Dataset):
    def __init__(self, data_dir, sub_list, sample_size=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_dir = data_dir
        self.subjects = sub_list
        self.sample_size = sample_size
        self.device = device  # שמירת ה-device כמאפיין

        self.all_samples = []
        self.all_targets = []

        for sub in sub_list:
            sub_dir = os.path.join(data_dir, sub)
            pkl_files = [f for f in os.listdir(sub_dir) if f.endswith('.pkl')]
            if len(pkl_files) != 13:
                print(f"אזהרה: נמצאו {len(pkl_files)} קבצים בתיקייה {sub_dir}, צפוי ל-13")
                continue

            for pkl_file in pkl_files:
                file_path = os.path.join(sub_dir, pkl_file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)  # [300, 143]
                data = data.T  # [143, 300]

                num_samples = data.shape[0] - sample_size  # 113
                for i in range(num_samples):
                    window = data[i:i + sample_size, :]  # [30, 300]
                    target = data[i + sample_size, :]  # [300]
                    self.all_samples.append(window)
                    self.all_targets.append(target)

        self.all_samples = np.stack(self.all_samples, axis=0)  # [1469 * len(sub_list), 30, 300]
        self.all_targets = np.stack(self.all_targets, axis=0)  # [1469 * len(sub_list), 300]


    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        time_series = torch.tensor(self.all_samples[idx], dtype=torch.float64, device=self.device)  # ישירות על ה-device
        time_point = torch.tensor(self.all_targets[idx], dtype=torch.float64, device=self.device)  # ישירות על ה-device
        return time_series, time_point


def file_to_sub_split():
    # Define paths
    input_folder = "D:\Final Project\Yuval_Gal\HCP_DATA"  # Change to your input folder path (e.g., 'C:/data/input/')
    output_base_path = "D:\Final Project\Yuval_Gal\New_folder"  # Change to your output directory path (e.g., 'C:/data/output/')

    # Ensure the output directory exists
    os.makedirs(output_base_path, exist_ok=True)

    # List all pkl files in the input folder
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]

    # Track progress
    total_files = len(pkl_files)
    processed_files = 0

    # Process each file
    for pkl_file in pkl_files:
        try:
            input_path = os.path.join(input_folder, pkl_file)

            # Load the file
            with open(input_path, 'rb') as file:
                df = pickle.load(file)

            # Filter out rows where is_rest = 1
            df = df[df['is_rest'] != 1]

            # Split by subject
            grouped = df.groupby('Subject')

            for subject_value, group_df in grouped:
                # Create a folder for the subject
                subject_folder = os.path.join(output_base_path, str(subject_value))
                os.makedirs(subject_folder, exist_ok=True)

                # Save the file with the same name in the appropriate folder
                output_file_path = os.path.join(subject_folder, pkl_file)

                with open(output_file_path, 'wb') as file:
                    pickle.dump(group_df, file)

            processed_files += 1
            print(f"Processed file {processed_files}/{total_files}: {pkl_file} completed")

        except Exception as e:
            print(f"Error processing file {pkl_file}: {str(e)}")
            continue

    print(f"Processing completed! Processed {processed_files} out of {total_files} files.")


def creat_data_Avg():
    # נתיבים
    source_root = "D:\Final Project\Yuval_Gal\New_folder"  # החלף בנתיב התיקייה הראשית המקורית
    destination_root = "D:\Final Project\Yuval_Gal\Avg_Data"  # החלף בנתיב היעד

    # יצירת התיקייה הראשית ביעד רק אם היא לא קיימת
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)
        print(f"Created destination root directory: {destination_root}")
    else:
        print(f"Destination root directory already exists: {destination_root}")

    # מעבר על כל התיקיות והקבצים במבנה הנתונים
    for root, dirs, files in os.walk(source_root):
        # יצירת הנתיב החדש תוך שמירה על מבנה התיקיות
        relative_path = os.path.relpath(root, source_root)  # הנתיב היחסי מהתיקייה הראשית
        destination_dir = os.path.join(destination_root, relative_path)

        # יצירת התיקייה ביעד אם היא לא קיימת
        os.makedirs(destination_dir, exist_ok=True)

        for file in files:
            if file.endswith(".pkl"):  # מתייחס רק לקובצי pkl
                source_file = os.path.join(root, file)

                # שינוי שם הקובץ (לדוגמה הוספת "_new" לשם הקובץ)
                new_file_name = f"{os.path.splitext(file)[0]}_Avg.pkl"
                destination_file = os.path.join(destination_dir, new_file_name)

                # קריאת קובץ ה-pkl
                with open(source_file, "rb") as f:
                    data = pickle.load(f)

                # כאן ניתן לשנות את הנתונים אם רוצים (data = process_data(data))
                #voxel_data = data.iloc[:, :-3]
                data["Voxels_Avg"] = data.iloc[:, :-4].mean(axis=1)
                print(data.iloc[:, -5:])

                # שמירת הקובץ החדש
                with open(destination_file, "wb") as f:
                    pickle.dump(data.iloc[:, -5:], f)

                print(f"Processed: {source_file} → {destination_file}")


def creat_movie_Mat():
    # נתיב לתיקייה הראשית של הנבדקים
    source_root = "D:\Final Project\Yuval_Gal\Avg_Data"  # החלף בנתיב המתאים
    destination_root = "D:\Final Project\Yuval_Gal\Processed_Matrices"  # תיקיית יעד למטריצות המעובדות

    # רשימת הסרטים לכלול
    movies_to_include = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    max_timepoints = 143  # הגבלה ל-143 timepoints

    # מעבר על כל תיקיות הנבדקים
    for subject_id in os.listdir(source_root):
        subject_path = os.path.join(source_root, subject_id)

        if os.path.isdir(subject_path):  # רק אם זו תיקייה
            # יצירת תיקייה חדשה לנבדק ביעד
            subject_dest_path = os.path.join(destination_root, subject_id)
            os.makedirs(subject_dest_path, exist_ok=True)

            # יצירת מילון לאחסון המטריצות של הנבדק
            matrices = {}

            # מעבר על כל הסרטים ברשימה
            for movie in movies_to_include:
                movie_matrix = []

                # מעבר על כל קבצי pkl בתוך תיקיית הנבדק
                for file in os.listdir(subject_path):
                    if file.endswith(".pkl"):  # מתייחס רק לקובצי pkl
                        file_path = os.path.join(subject_path, file)

                        # קריאת קובץ ה-pkl
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)

                        # בדיקה שהעמודות קיימות
                        if 'y' in data and 'Voxels_Avg' in data:
                            # סינון לפי מספר הסרט
                            movie_avg = data['Voxels_Avg'][data['y'] == movie]

                            # הגבלת מספר נקודות הזמן
                            if movie_avg.shape[0] > max_timepoints:
                                movie_avg = movie_avg.iloc[:max_timepoints]

                            movie_matrix.append(movie_avg)

                # אם נמצאו נתונים עבור הסרט, נשמור את המטריצה
                if movie_matrix:
                    movie_matrix_np = np.array(movie_matrix)  # המרת הרשימה למערך NumPy
                    matrices[movie] = movie_matrix_np

                    # שמירת המטריצה כקובץ pkl בתיקיית היעד
                    movie_filename = f"movie_{movie}.pkl"
                    movie_filepath = os.path.join(subject_dest_path, movie_filename)

                    with open(movie_filepath, "wb") as f:
                        pickle.dump(movie_matrix_np, f)

                    print(f"Saved {movie_filepath} with shape {movie_matrix_np.shape}")

    print("Processing complete.")