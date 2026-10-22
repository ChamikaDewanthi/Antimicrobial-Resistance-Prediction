import pandas as pd

from get_file import get_main_file
from get_file import get_file
from clean_data import clean_data
from clean_data import remove_duplicates
from clean_data import remove_identifiers

def prepare_dataset():
    try:
        # Dataset Source:
        # Antibiotic Resistance Microbiology Dataset (ARMD)
        # Dataset DOI: https://doi.org/10.5061/dryad.jq2bvq8kp
        # Publication: https://doi.org/10.1038/s41597-025-05649-7

        # Loads the main file
        print("Loading main dataset...")
        main_columns = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "ordering_mode", "culture_description", "organism", "antibiotic", "susceptibility"]
        main_file = get_main_file("data/microbiology_cultures_cohort.csv", main_columns)
        print("\033[92mMain dataset is successfully loaded\033[0m")

        file_names = []
        file_names.append("demographics")
        file_names.append("prior_infecting_organism")
        #file_names.append("labs")
        file_names.append("vitals")
        file_names.append("antibiotic_class_exposure")

        file_paths = []
        file_paths.append("data/microbiology_cultures_demographics.csv")
        file_paths.append("data/microbiology_culture_prior_infecting_organism.csv")
        #file_paths.append("data/microbiology_cultures_labs.csv")
        file_paths.append("data/microbiology_cultures_vitals.csv")
        file_paths.append("data/microbiology_cultures_antibiotic_class_exposure.csv")

        columns_list = []
        columns_list.append(["anon_id", "age", "gender"])
        columns_list.append(["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "prior_organism", "prior_infecting_organism_days_to_culutre"])
        #columns_list.append(["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "median_wbc", "median_neutrophils", "median_lactate", "median_cr"])
        columns_list.append(["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "median_heartrate", "median_resprate", "median_temp"])
        columns_list.append(["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "antibiotic_class", "time_to_culturetime"])

        # Loads data from files and merge to the main dataset by primary keys
        primary_keys = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]
        print("Loading essential characteristics from other datasets...")
        for i in range (0, len(file_paths)):            
            file = get_file(file_paths[i], columns_list[i], file_names[i])
            if(file_names[i] == "demographics"):
                main_file = pd.merge(main_file, file, on = "anon_id", how = "left")
            else:
                main_file = pd.merge(main_file, file, on = primary_keys, how = "left")
            del file
        print("\033[92mDatasets are successfully merged\033[0m")

        # Handles missing values
        print("Cleaning processed dataset...")
        main_file = clean_data(main_file)

        # Saves merged file in results directory
        print("Saving processed dataset...")
        main_file.to_csv("results/merged_dateset.csv", index = False)
        main_file = remove_identifiers(main_file)
        main_file = remove_duplicates(main_file)
        main_file.to_csv("results/final_dateset.csv", index = False)
        print("\033[92mProcessed dataset is successfully saved\033[0m")

    except Exception as e:
        print(f"Unexpected error: {e}")