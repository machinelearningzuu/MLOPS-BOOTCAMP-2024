import pytest
from src.pipeline.prediction_pipeline import *

ppipe = PredictPipeline()

@pytest.fixture
def single_prediction(test_file = './data/raw/test.csv'):
    test_dataset = pd.read_csv(test_file)
    single_json = eval(test_dataset[:1].to_json(orient='records'))[0]
    single_data = CustomData(single_json)
    single_df = single_data.get_data_as_dataframe()
    result = ppipe.predict(single_df)
    return result

def test_single_pred_not_none(single_prediction): # output is not none
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is string
    assert isinstance(single_prediction,str)

def test_single_pred_validate(single_prediction): # check the output is Y
    assert single_prediction == 'Y'