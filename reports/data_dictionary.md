# Data dictionary — `data/features.parquet`

495,532 rows, 36 model inputs + target. Fitted encodings (last column) are applied by `src/features.py::make_preprocessor()` on the training set only.

Extra fitted feature: **location_cluster** = KMeans (k = 50, seed 42) on Start_Lat / Start_Lng, one-hot encoded.

Not used as inputs: the 5 columns dropped in feature selection (end of file), Street (-> is_highway), Weather_Condition (-> weather_group), Zipcode (too many values; City/County cover location), Start_Time (-> hour/weekday/month/year), light columns (-> *_night flags).

| column | dtype | built_from | description | pipeline_encoding |
| --- | --- | --- | --- | --- |
| Start_Lat | float64 | raw | Latitude of the accident start (°) | median impute (+ scale for LogReg) |
| Start_Lng | float64 | raw | Longitude of the accident start (°) | median impute (+ scale for LogReg) |
| Temperature(F) | float64 | raw | Air temperature (°F); impossible values set to NaN in Phase 4 | median impute (+ scale for LogReg) |
| Humidity(%) | float64 | raw | Relative humidity (%) | median impute (+ scale for LogReg) |
| Pressure(in) | float64 | raw | Air pressure (inHg) | median impute (+ scale for LogReg) |
| Visibility(mi) | float64 | raw | Visibility (miles) | median impute (+ scale for LogReg) |
| Wind_Speed(mph) | float64 | raw | Wind speed (mph) | median impute (+ scale for LogReg) |
| Precipitation(in) | float64 | raw | Precipitation (inches); missing -> 0 in Phase 4 | median impute (+ scale for LogReg) |
| precip_missing | int8 | Phase 4 | 1 if Precipitation was missing and filled with 0 | median impute (+ scale for LogReg) |
| hour | int8 | Start_Time | Hour of day, local time (0-23) | median impute (+ scale for LogReg) |
| weekday | int8 | Start_Time | Day of week (0 = Monday ... 6 = Sunday) | median impute (+ scale for LogReg) |
| month | int8 | Start_Time | Month (1-12) | median impute (+ scale for LogReg) |
| year | int16 | Start_Time | Year (2016-2023) | median impute (+ scale for LogReg) |
| is_weekend | int8 | weekday | 1 if Saturday or Sunday | median impute (+ scale for LogReg) |
| is_rush_hour | int8 | weekday, hour | 1 if weekday and hour in 7-9 or 16-19 | median impute (+ scale for LogReg) |
| civil_night | int8 | Civil_Twilight | 1 if Night by civil twilight | median impute (+ scale for LogReg) |
| astro_night | int8 | Astronomical_Twilight | 1 if Night by astronomical twilight | median impute (+ scale for LogReg) |
| bad_weather | int8 | weather_group | 1 if Rain, Heavy rain / Thunderstorm, Snow / Ice or Fog / Haze / Smoke | median impute (+ scale for LogReg) |
| road_feature_count | int8 | road features | Number of road features near the accident (0-12) | median impute (+ scale for LogReg) |
| is_highway | int8 | Street | 1 if street name looks like a highway (I-, US-, state route XX-, Interstate, Hwy, Highway, Fwy, Expy, Tpke, Beltway, State Pkwy, Route...) | median impute (+ scale for LogReg) |
| Amenity | int8 | raw | 1 if a amenity is near the accident | median impute (+ scale for LogReg) |
| Crossing | int8 | raw | 1 if a crossing is near the accident | median impute (+ scale for LogReg) |
| Give_Way | int8 | raw | 1 if a give way is near the accident | median impute (+ scale for LogReg) |
| Junction | int8 | raw | 1 if a junction is near the accident | median impute (+ scale for LogReg) |
| No_Exit | int8 | raw | 1 if a no exit is near the accident | median impute (+ scale for LogReg) |
| Railway | int8 | raw | 1 if a railway is near the accident | median impute (+ scale for LogReg) |
| Station | int8 | raw | 1 if a station is near the accident | median impute (+ scale for LogReg) |
| Stop | int8 | raw | 1 if a stop is near the accident | median impute (+ scale for LogReg) |
| Traffic_Signal | int8 | raw | 1 if a traffic signal is near the accident | median impute (+ scale for LogReg) |
| State | str | raw | US state (49 values) | one-hot |
| Source | str | raw | Data provider (Source1-3); records severity differently (EDA chart 12) | one-hot |
| weather_group | str | Weather_Condition | 9 groups: Clear, Cloudy, Rain, Heavy rain / Thunderstorm, Snow / Ice, Fog / Haze / Smoke, Windy, Other, Unknown | one-hot |
| Wind_Direction | str | raw | 16 compass points + CALM + VAR + Unknown | one-hot |
| season | str | month | Winter (Dec-Feb), Spring, Summer, Autumn | one-hot |
| City | str | raw | City (~9,500 values) | frequency (train shares) |
| County | str | raw | County (~1,600 values) | frequency (train shares) |
| Severity | int8 | raw | TARGET: impact on traffic, 1 (least) to 4 (most). Not injuries. | label |

## Dropped during feature selection (notebook 04)

| column | reason |
| --- | --- |
| Roundabout | near-constant (1 in < 0.1% of rows); still counted in road_feature_count |
| Bump | near-constant (1 in < 0.1% of rows); still counted in road_feature_count |
| Traffic_Calming | near-constant (1 in < 0.1% of rows); still counted in road_feature_count |
| is_night | redundant: |r| = 0.89 with civil_night, which has higher mutual information |
| nautical_night | redundant: |r| = 0.88 with astro_night, which has higher mutual information |
