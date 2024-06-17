-- Create table for df1 data
CREATE TABLE df1 (
    id SERIAL PRIMARY KEY,
    order_id INT,
    driver_id INT,
    driver_action TEXT,
    lat FLOAT,
    lng FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Create table for df2 data
CREATE TABLE df2 (
    trip_id INT PRIMARY KEY,
    trip_origin TEXT,
    trip_destination TEXT,
    trip_start_time TIMESTAMP,
    trip_end_time TIMESTAMP
);
