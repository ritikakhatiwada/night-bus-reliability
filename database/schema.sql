-- ─────────────────────────────────────────────────────────────────────────────
-- schema.sql  –  Night Bus Service Reliability Prediction System
-- ST5011CEM | Big Data Programming Project
-- ─────────────────────────────────────────────────────────────────────────────

-- Table 1: Cleaned & feature-engineered timetable records
CREATE TABLE IF NOT EXISTS timetables_clean (
    trip_id                  TEXT    PRIMARY KEY,
    service_code             TEXT    NOT NULL,   -- e.g. N1, N14
    operator                 TEXT    NOT NULL,
    scheduled_depart         TEXT,               -- ISO-8601 datetime
    actual_depart            TEXT,
    delay_minutes            REAL,               -- negative = early
    delay_minutes_clipped    REAL,               -- clipped -30 to +60
    is_delayed               INTEGER NOT NULL,   -- 1=delayed, 0=on-time (±2 min)
    weather                  TEXT,
    weather_code             INTEGER,            -- 0=Clear … 4=Snow
    scheduled_duration_min   INTEGER,
    actual_duration_min      INTEGER,
    passenger_count          INTEGER,
    capacity                 INTEGER,
    load_factor              REAL,               -- passenger_count / capacity
    vehicle_age_years        INTEGER,
    driver_experience_yrs    INTEGER,
    hour_of_night            INTEGER,
    day_of_week              TEXT,
    disruption_count         INTEGER DEFAULT 0,
    high_severity_count      INTEGER DEFAULT 0,
    mean_dur                 REAL,
    std_dur                  REAL,
    cv_travel_time           REAL,               -- Coefficient of Variation
    pct_on_time              REAL,               -- Service Reliability metric
    headway_min              REAL
);

-- Table 2: Disruption events
CREATE TABLE IF NOT EXISTS disruptions_clean (
    disruption_id    TEXT    PRIMARY KEY,
    service_code     TEXT    NOT NULL,
    operator         TEXT,
    disruption_type  TEXT,
    start_time       TEXT,
    duration_min     INTEGER,
    severity         TEXT    CHECK(severity IN ('Low','Medium','High')),
    affected_stops   INTEGER,
    resolved         INTEGER                     -- 1=True, 0=False
);

-- Table 3: GPS locations enriched with trip info
CREATE TABLE IF NOT EXISTS locations_enriched (
    location_id      TEXT    PRIMARY KEY,
    trip_id          TEXT    REFERENCES timetables_clean(trip_id),
    timestamp        TEXT,
    latitude         REAL,
    longitude        REAL,
    speed_kmh        REAL,
    heading_deg      REAL,
    signal_quality   TEXT,
    service_code     TEXT,
    operator         TEXT,
    is_delayed       INTEGER
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_tt_service   ON timetables_clean  (service_code);
CREATE INDEX IF NOT EXISTS idx_tt_operator  ON timetables_clean  (operator);
CREATE INDEX IF NOT EXISTS idx_tt_delayed   ON timetables_clean  (is_delayed);
CREATE INDEX IF NOT EXISTS idx_tt_hour      ON timetables_clean  (hour_of_night);
CREATE INDEX IF NOT EXISTS idx_dis_service  ON disruptions_clean (service_code);
CREATE INDEX IF NOT EXISTS idx_loc_trip     ON locations_enriched (trip_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Sample parameterised queries (no string concatenation – SQL injection safe)
-- In Python: conn.execute(text("SELECT … WHERE operator = :op"), {"op": value})
-- ─────────────────────────────────────────────────────────────────────────────

-- Q1: Service Reliability per operator
-- SELECT operator,
--        ROUND(1.0 - AVG(is_delayed), 4)  AS service_reliability,
--        COUNT(*)                           AS total_trips
-- FROM timetables_clean
-- GROUP BY operator
-- ORDER BY service_reliability DESC;

-- Q2: Routes failing the 85% on-time threshold
-- SELECT service_code,
--        ROUND(AVG(CASE WHEN is_delayed=0 THEN 1.0 ELSE 0.0 END)*100, 2) AS pct_on_time,
--        COUNT(*) AS trips
-- FROM timetables_clean
-- GROUP BY service_code
-- HAVING pct_on_time < 85.0
-- ORDER BY pct_on_time;

-- Q3: Disruption hotspots
-- SELECT service_code, disruption_type, COUNT(*) AS n
-- FROM disruptions_clean
-- WHERE severity = 'High'
-- GROUP BY service_code, disruption_type
-- ORDER BY n DESC
-- LIMIT 20;

-- Q4: Travel Time Variability (CV) by route
-- SELECT service_code,
--        ROUND(AVG(cv_travel_time), 4) AS mean_cv
-- FROM timetables_clean
-- GROUP BY service_code
-- HAVING mean_cv > 0.15
-- ORDER BY mean_cv DESC;
