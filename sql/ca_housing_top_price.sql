-- Top 20 highest median house value districts in the CA housing table
SELECT *
FROM CA_housing
ORDER BY median_house_value DESC
LIMIT 20;
