import os

class Config:
    DF1_PATH: str = "/home/moraa/Documents/10_academy/Week-8/Data/driver_locations_during_request.csv"
    DF2_PATH: str = "/home/moraa/Documents/10_academy/Week-8/Data/nb.csv"
    DF1_DROP_COLUMNS: list = ['created_at', 'updated_at']
    DF2_IMPUTE_COLUMNS: dict = {
        'Trip Start Time': 'mode',
        'Trip End Time': 'mode'
    }
    LOG_FILE: str = 'logs/preprocessing.log'


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
    LOGS_DIR = os.path.join(BASE_DIR, '..', '..', 'logs')

    DF1_PATH = os.path.join(DATA_DIR, 'driver_locations_during_request.csv')
    DF2_PATH = os.path.join(DATA_DIR, 'nb.csv')
    LOG_FILE = os.path.join(LOGS_DIR, 'preprocessing.log')

    HOLIDAYS_2021 = [
        '2021-01-01', '2021-04-02', '2021-04-05', '2021-05-01', '2021-05-12', '2021-05-13',
        '2021-06-12', '2021-07-20', '2021-07-21', '2021-10-01', '2021-10-18', '2021-10-19',
        '2021-12-25', '2021-12-26'
    ]

    HOLIDAYS_2022 = [
        '2022-01-01', '2022-01-03', '2022-04-15', '2022-04-18', '2022-05-01', '2022-05-02',
        '2022-05-03', '2022-05-04', '2022-06-12', '2022-06-13', '2022-07-09', '2022-07-10',
        '2022-07-11', '2022-07-12', '2022-10-01', '2022-10-03', '2022-10-08', '2022-10-09',
        '2022-10-10', '2022-12-25', '2022-12-26', '2022-12-27'
    ]
    
    RADIUS = 0.5  # Radius in kilometers for counting riders around accepted orders
