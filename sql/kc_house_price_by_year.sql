-- Average sale price per year for kc_house_data
-- date column is stored as text like '20141013T000000'
SELECT
    CAST(substr(date, 1, 4) AS INTEGER) AS year,
    AVG(price) AS avg_price,
    COUNT(*) AS num_sales
FROM kc_house_data
GROUP BY year
ORDER BY year;
