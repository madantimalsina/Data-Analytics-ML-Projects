-- Class balance for the breast cancer classification dataset
SELECT
    benign_0__mal_1 AS label,
    COUNT(*) AS num_samples
FROM cancer_classification
GROUP BY benign_0__mal_1;
