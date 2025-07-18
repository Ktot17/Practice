import time

MAX_FILE_SIZE_MB = 50
PERIODIC_CLEANUP_INTERVAL_SECONDS = 300
SESSION_TIMEOUT_SECONDS = 3600

MODEL_URL = "/download-model"
MODEL_FILE_EXT = ".joblib"
OUT_DIR = "output/"


class DataColumns:
    temperature_col_name = "Температура_печи_C"
    humidity_col_name = "Влажность_глины_%"
    pressure_col_name = "Давление_пресса_МПа"
    brand_col_name = "Марка_глины"
    operator_col_name = "Оператор_смены"
    defect_col_name = "Брак_на_1000_шт"


class ResultData:
    mae = "MAE"
    r2 = "R2"
    str_amount = "Количество строк"
    features = "Признаки"
    missing = "Пропуски"
    missing_by_cols = "Пропуски по столбцам"
    pred_graph_path = "График предсказаний"
    feat_graph_path = "График важности признаков"
    model_url = "URL для загрузки .joblib файла"


class Session:
    def __init__(self):
        self._data = None
        self._result = None
        self.last_activity = time.time()

    @property
    def data(self):
        self.last_activity = time.time()
        return self._data

    @data.setter
    def data(self, value):
        self.last_activity = time.time()
        self._data = value

    @property
    def result(self):
        self.last_activity = time.time()
        return self._result

    @result.setter
    def result(self, value):
        self.last_activity = time.time()
        self._result = value
