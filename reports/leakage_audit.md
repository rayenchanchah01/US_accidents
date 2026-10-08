# Phase 3 — Leakage audit

Rule: a column is kept only if its value is known when the accident is first reported.

| column | decision | reason | n_unique | missing_% |
| --- | --- | --- | --- | --- |
| ID | drop | Identifier only, no information | 500000 | 0.0 |
| Source | keep (analysis) | Providers may record severity differently; with/without ablation in Phase 9 | 3 | 0.0 |
| Severity | keep | Known at report time (target) | 4 | 0.0 |
| Start_Time | keep | Known at report time (time) | 487027 | 0.0 |
| End_Time | drop | Leakage: known only after the road is cleared (duration ≈ delay ≈ severity) | 493821 | 0.0 |
| Start_Lat | keep | Known at report time (location) | 368965 | 0.0 |
| Start_Lng | keep | Known at report time (location) | 370195 | 0.0 |
| End_Lat | drop | Leakage: end of the affected stretch, measured afterwards | 220333 | 44.08 |
| End_Lng | drop | Leakage: end of the affected stretch, measured afterwards | 221463 | 44.08 |
| Distance(mi) | drop | Leakage: length of road affected is part of how severity is judged | 10627 | 0.0 |
| Description | drop | Leakage: free text says 'road closed' / 'lanes blocked' | 408146 | 0.0 |
| Street | keep (care) | Keep with care: high cardinality, encode in Phase 6 (no one-hot) | 88272 | 0.14 |
| City | keep (care) | Keep with care: high cardinality, encode in Phase 6 (no one-hot) | 9488 | 0.0 |
| County | keep (care) | Keep with care: high cardinality, encode in Phase 6 (no one-hot) | 1606 | 0.0 |
| State | keep | Known at report time (location) | 49 | 0.0 |
| Zipcode | keep (care) | Keep with care: high cardinality, encode in Phase 6 (no one-hot) | 127804 | 0.02 |
| Country | drop | Constant (always 'US') | 1 | 0.0 |
| Timezone | drop | Redundant with location (State) | 4 | 0.1 |
| Airport_Code | drop | Redundant with location (nearest weather station) | 1878 | 0.29 |
| Weather_Timestamp | drop | Redundant with Start_Time | 255623 | 1.53 |
| Temperature(F) | keep | Known at report time (weather) | 712 | 2.09 |
| Wind_Chill(F) | keep | Known at report time (weather) | 797 | 25.8 |
| Humidity(%) | keep | Known at report time (weather) | 100 | 2.23 |
| Pressure(in) | keep | Known at report time (weather) | 980 | 1.79 |
| Visibility(mi) | keep | Known at report time (weather) | 65 | 2.26 |
| Wind_Direction | keep | Known at report time (weather) | 24 | 2.24 |
| Wind_Speed(mph) | keep | Known at report time (weather) | 95 | 7.4 |
| Precipitation(in) | keep | Known at report time (weather) | 175 | 28.52 |
| Weather_Condition | keep | Known at report time (weather) | 108 | 2.22 |
| Amenity | keep | Known at report time (road feature) | 2 | 0.0 |
| Bump | keep | Known at report time (road feature) | 2 | 0.0 |
| Crossing | keep | Known at report time (road feature) | 2 | 0.0 |
| Give_Way | keep | Known at report time (road feature) | 2 | 0.0 |
| Junction | keep | Known at report time (road feature) | 2 | 0.0 |
| No_Exit | keep | Known at report time (road feature) | 2 | 0.0 |
| Railway | keep | Known at report time (road feature) | 2 | 0.0 |
| Roundabout | keep | Known at report time (road feature) | 2 | 0.0 |
| Station | keep | Known at report time (road feature) | 2 | 0.0 |
| Stop | keep | Known at report time (road feature) | 2 | 0.0 |
| Traffic_Calming | keep | Known at report time (road feature) | 2 | 0.0 |
| Traffic_Signal | keep | Known at report time (road feature) | 2 | 0.0 |
| Turning_Loop | drop | Constant (single value) | 1 | 0.0 |
| Sunrise_Sunset | keep | Known at report time (light) | 2 | 0.3 |
| Civil_Twilight | keep | Known at report time (light) | 2 | 0.3 |
| Nautical_Twilight | keep | Known at report time (light) | 2 | 0.3 |
| Astronomical_Twilight | keep | Known at report time (light) | 2 | 0.3 |
