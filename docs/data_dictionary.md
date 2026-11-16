# Model inputs

All features are row-local except training-fitted encoding. Source and outcome fields are excluded.

| Feature | Type |
|---|---|
| Start_Lat | Numeric; missing values retained |
| Start_Lng | Numeric; missing values retained |
| Temperature(F) | Numeric; missing values retained |
| Humidity(%) | Numeric; missing values retained |
| Pressure(in) | Numeric; missing values retained |
| Visibility(mi) | Numeric; missing values retained |
| Wind_Speed(mph) | Numeric; missing values retained |
| Precipitation(in) | Numeric; missing values retained |
| Amenity | Numeric; missing values retained |
| Bump | Numeric; missing values retained |
| Crossing | Numeric; missing values retained |
| Give_Way | Numeric; missing values retained |
| Junction | Numeric; missing values retained |
| No_Exit | Numeric; missing values retained |
| Railway | Numeric; missing values retained |
| Roundabout | Numeric; missing values retained |
| Station | Numeric; missing values retained |
| Stop | Numeric; missing values retained |
| Traffic_Calming | Numeric; missing values retained |
| Traffic_Signal | Numeric; missing values retained |
| Year | Numeric; missing values retained |
| Month | Numeric; missing values retained |
| Hour | Numeric; missing values retained |
| Weekday | Numeric; missing values retained |
| Is_Weekend | Numeric; missing values retained |
| Is_Rush_Hour | Numeric; missing values retained |
| Is_Highway | Numeric; missing values retained |
| Road_Feature_Count | Numeric; missing values retained |
| Weather_Unavailable | Numeric; missing values retained |
| Precipitation_Missing | Numeric; missing values retained |
| State | Native categorical |
| City | Native categorical |
| County | Native categorical |
| Weather_Group | Native categorical |
| Wind_Direction | Native categorical |
| Sunrise_Sunset | Native categorical |

## Excluded columns

| Column | Reason |
|---|---|
| End_Time | Known after the accident: outcome leakage. |
| End_Lat | End-of-incident extent: outcome leakage. |
| End_Lng | End-of-incident extent: outcome leakage. |
| Distance(mi) | Reported affected extent may encode outcome. |
| Description | Narrative can describe outcome and closure. |
| ID | Identifier, retained only for traceability. |
| Source | Reporting-provider shortcut; retained only for auditing. |
| Weather_Timestamp | Used only to reject future/stale observations. |
| Wind_Chill(F) | Redundant derived temperature, often absent. |
| Zipcode | Redundant high-cardinality location identifier. |
| Street | Only a deterministic highway flag is retained. |
| Country | Constant in this US dataset. |
| Timezone | Local time and state already represented. |
| Airport_Code | Weather-station proxy, unnecessary location shortcut. |
| Turning_Loop | Uninformative road feature in the supplied dataset. |
| Civil_Twilight | Redundant light indicator. |
| Nautical_Twilight | Redundant light indicator. |
| Astronomical_Twilight | Redundant light indicator. |
