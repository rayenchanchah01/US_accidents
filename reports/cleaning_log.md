# Phase 4 — Cleaning log

## Steps (shape after each step)

| step | rows | Δrows | cols | Δcols | note |
| --- | --- | --- | --- | --- | --- |
| Load sample.parquet | 500000 | 0 | 46 | 0 |  |
| Phase 3: drop leakage / ID / constant / redundant columns | 500000 | 0 | 35 | -11 | ID, End_Time, End_Lat, End_Lng, Distance(mi), Description, Country, Timezone, Airport_Code, Weather_Timestamp, Turning_Loop |
| 4.1 Parse Start_Time; road flags True/False -> 0/1 | 500000 | 0 | 35 | 0 |  |
| 4.1 Drop exact duplicate rows | 497116 | -2884 | 35 | 0 | 2884 removed |
| 4.2 Drop columns >40% missing + Wind_Chill(F) | 497116 | 0 | 34 | -1 | Wind_Chill(F) |
| 4.2 Precipitation NaN -> 0 + precip_missing flag | 497116 | 0 | 35 | 1 | 142481 filled |
| 4.2 Drop rows missing City / Zipcode / light columns | 495532 | -1584 | 35 | 0 | 1584 rows removed |
| 4.2 Weather_Condition, Wind_Direction NaN -> 'Unknown' | 495532 | 0 | 35 | 0 |  |
| 4.3 Impossible weather values -> NaN | 495532 | 0 | 35 | 0 | Temperature(F): 4, Wind_Speed(mph): 3, Visibility(mi): 3, Pressure(in): 3 |
| 4.4 Wind_Direction -> 16 compass points + CALM + VAR | 495532 | 0 | 35 | 0 | 19 values |
| 4.4 Weather_Condition -> Weather_Group | 495532 | 0 | 36 | 1 | 109 labels -> 9 groups |
| Save clean.parquet | 495532 | 0 | 36 | 0 |  |

## Missing values before 4.2 (%)

| column | missing_% |
| --- | --- |
| Precipitation(in) | 28.66 |
| Wind_Chill(F) | 25.93 |
| Wind_Speed(mph) | 7.42 |
| Visibility(mi) | 2.26 |
| Wind_Direction | 2.24 |
| Humidity(%) | 2.23 |
| Weather_Condition | 2.22 |
| Temperature(F) | 2.09 |
| Pressure(in) | 1.79 |
| Civil_Twilight | 0.3 |
| Nautical_Twilight | 0.3 |
| Astronomical_Twilight | 0.3 |
| Sunrise_Sunset | 0.3 |
| Street | 0.14 |
| Zipcode | 0.02 |

## Impossible-value thresholds (4.3)

| column | low | high | set_to_NaN | reason |
| --- | --- | --- | --- | --- |
| Temperature(F) | -50 | 130 | 4 | US record high is 134 °F; below −50 °F is not plausible on US roads in this data |
| Wind_Speed(mph) | 0 | 200 | 3 | Above the strongest surface winds ever measured outside tornadoes |
| Visibility(mi) | 0 | 100 | 3 | Weather stations do not report visibility above ~100 mi |
| Pressure(in) | 15 | 32 | 3 | Highest US roads (~14,000 ft) ≈ 17.5 inHg; world record high 32.06 inHg |

## Wind_Direction merges (4.4)

| raw | standard |
| --- | --- |
| Calm | CALM |
| Variable | VAR |
| North | N |
| South | S |
| East | E |
| West | W |

## Weather_Condition -> Weather_Group (4.4)

| Weather_Group | rows | labels |
| --- | --- | --- |
| Clear | 216488 | Fair, Clear |
| Cloudy | 200063 | Mostly Cloudy, Cloudy, Partly Cloudy, Overcast, Scattered Clouds |
| Rain | 30629 | Light Rain, Rain, Light Drizzle, Light Rain / Windy, Drizzle, Rain / Windy, Showers in the Vicinity, Heavy Drizzle, Drizzle and Fog, Light Rain Shower, Light Drizzle / Windy, Light Rain Showers, Rain Showers, Light Rain Shower / Windy, Rain Shower, Drizzle / Windy |
| Fog / Haze / Smoke | 13084 | Fog, Haze, Smoke, Patches of Fog, Mist, Shallow Fog, Haze / Windy, Light Freezing Fog, Fog / Windy, Blowing Dust / Windy, Blowing Dust, Widespread Dust, Smoke / Windy, Sand / Dust Whirlwinds, Widespread Dust / Windy, Partial Fog, Light Haze, Partial Fog / Windy, Volcanic Ash, Mist / Windy, Sand |
| Snow / Ice | 11278 | Light Snow, Snow, Wintry Mix, Light Snow / Windy, Heavy Snow, Light Freezing Rain, Light Freezing Drizzle, Snow / Windy, Blowing Snow / Windy, Heavy Snow / Windy, Blowing Snow, Snow and Sleet, Light Sleet, Wintry Mix / Windy, Light Ice Pellets, Freezing Rain, Light Snow and Sleet, Ice Pellets, Sleet, Light Freezing Rain / Windy, Light Snow Shower, Snow and Sleet / Windy, Light Snow and Sleet / Windy, Freezing Drizzle, Light Snow Showers, Sleet / Windy, Heavy Freezing Drizzle, Heavy Sleet, Snow Grains, Light Snow Grains, Drifting Snow / Windy, Light Blowing Snow, Low Drifting Snow |
| Unknown | 10750 | Unknown |
| Heavy rain / Thunderstorm | 7882 | Heavy Rain, Thunder in the Vicinity, T-Storm, Thunder, Light Rain with Thunder, Heavy T-Storm, Light Thunderstorms and Rain, Thunderstorm, Heavy Thunderstorms and Rain, Thunderstorms and Rain, Heavy Rain / Windy, Heavy T-Storm / Windy, T-Storm / Windy, Thunder / Windy, Small Hail, Funnel Cloud, Hail, Thunder / Wintry Mix, Light Snow with Thunder, Light Thunderstorms and Snow, Light Hail, Snow and Thunder, Tornado, Thunder and Hail |
| Windy | 5155 | Fair / Windy, Mostly Cloudy / Windy, Cloudy / Windy, Partly Cloudy / Windy, Squalls / Windy, Squalls |
| Other | 203 | N/A Precipitation |

## NaN left for Pipeline imputation (Phase 9)

| column | NaN |
| --- | --- |
| Street | 673 |
| Temperature(F) | 10129 |
| Humidity(%) | 10781 |
| Pressure(in) | 8597 |
| Visibility(mi) | 10942 |
| Wind_Speed(mph) | 36619 |
