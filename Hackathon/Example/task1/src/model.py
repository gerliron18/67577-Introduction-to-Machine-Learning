"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):

===================================================
"""

class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        self.__path_to_weather = path_to_weather

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        x['ArrDelay'] = x['CRSDepTime'] / 100 - 12
        late = x['ArrDelay'] > 0
        x.loc[late, 'DelayFactor'] = (x['ArrDelay'] // 3).astype(int)
        delay_factors = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']
        x = x.replace({'DelayFactor': {i: delay_factors[i] for i in range(4)}})
        return x[['ArrDelay', 'DelayFactor']]
