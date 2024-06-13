class Config:
    DF1_PATH: str = "/home/moraa/Documents/10_academy/Week-8/Data/driver_locations_during_request.csv"
    DF2_PATH: str = "/home/moraa/Documents/10_academy/Week-8/Data/nb.csv"
    DF1_DROP_COLUMNS: list = ['created_at', 'updated_at']
    DF2_IMPUTE_COLUMNS: dict = {
        'Trip Start Time': 'mode',
        'Trip End Time': 'mode'
    }
    LOG_FILE: str = 'logs/preprocessing.log'
