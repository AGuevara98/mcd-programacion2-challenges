-- CRITIC weighting for NP-RV indicators (PostgreSQL)
--
-- Expected source relation example:
--   features.station_nprv_audit(station_id, node_connectivity, place_population,
--                                place_employment, real_estate_proxy,
--                                vitality_retail, vitality_food, vitality_social)
--
-- Replace schema/table names as needed.

WITH base AS (
    SELECT
        station_id,
        node_connectivity::double precision AS node_connectivity,
        place_population::double precision AS place_population,
        place_employment::double precision AS place_employment,
        real_estate_proxy::double precision AS real_estate_proxy,
        vitality_retail::double precision AS vitality_retail,
        vitality_food::double precision AS vitality_food,
        vitality_social::double precision AS vitality_social
    FROM features.station_nprv_audit
),
std AS (
    SELECT
        stddev_pop(node_connectivity) AS sd_node_connectivity,
        stddev_pop(place_population) AS sd_place_population,
        stddev_pop(place_employment) AS sd_place_employment,
        stddev_pop(real_estate_proxy) AS sd_real_estate_proxy,
        stddev_pop(vitality_retail) AS sd_vitality_retail,
        stddev_pop(vitality_food) AS sd_vitality_food,
        stddev_pop(vitality_social) AS sd_vitality_social
    FROM base
),
corrs AS (
    SELECT
        corr(node_connectivity, place_population) AS c_np,
        corr(node_connectivity, place_employment) AS c_ne,
        corr(node_connectivity, real_estate_proxy) AS c_nr,
        corr(node_connectivity, vitality_retail) AS c_nvr,
        corr(node_connectivity, vitality_food) AS c_nvf,
        corr(node_connectivity, vitality_social) AS c_nvs,

        corr(place_population, place_employment) AS c_pe,
        corr(place_population, real_estate_proxy) AS c_pr,
        corr(place_population, vitality_retail) AS c_pvr,
        corr(place_population, vitality_food) AS c_pvf,
        corr(place_population, vitality_social) AS c_pvs,

        corr(place_employment, real_estate_proxy) AS c_er,
        corr(place_employment, vitality_retail) AS c_evr,
        corr(place_employment, vitality_food) AS c_evf,
        corr(place_employment, vitality_social) AS c_evs,

        corr(real_estate_proxy, vitality_retail) AS c_rvr,
        corr(real_estate_proxy, vitality_food) AS c_rvf,
        corr(real_estate_proxy, vitality_social) AS c_rvs,

        corr(vitality_retail, vitality_food) AS c_vrvf,
        corr(vitality_retail, vitality_social) AS c_vrvs,
        corr(vitality_food, vitality_social) AS c_vfvs
    FROM base
),
critic_scores AS (
    SELECT
        s.sd_node_connectivity * (
            (1 - COALESCE(c.c_np, 0)) +
            (1 - COALESCE(c.c_ne, 0)) +
            (1 - COALESCE(c.c_nr, 0)) +
            (1 - COALESCE(c.c_nvr, 0)) +
            (1 - COALESCE(c.c_nvf, 0)) +
            (1 - COALESCE(c.c_nvs, 0))
        ) AS score_node_connectivity,

        s.sd_place_population * (
            (1 - COALESCE(c.c_np, 0)) +
            (1 - COALESCE(c.c_pe, 0)) +
            (1 - COALESCE(c.c_pr, 0)) +
            (1 - COALESCE(c.c_pvr, 0)) +
            (1 - COALESCE(c.c_pvf, 0)) +
            (1 - COALESCE(c.c_pvs, 0))
        ) AS score_place_population,

        s.sd_place_employment * (
            (1 - COALESCE(c.c_ne, 0)) +
            (1 - COALESCE(c.c_pe, 0)) +
            (1 - COALESCE(c.c_er, 0)) +
            (1 - COALESCE(c.c_evr, 0)) +
            (1 - COALESCE(c.c_evf, 0)) +
            (1 - COALESCE(c.c_evs, 0))
        ) AS score_place_employment,

        s.sd_real_estate_proxy * (
            (1 - COALESCE(c.c_nr, 0)) +
            (1 - COALESCE(c.c_pr, 0)) +
            (1 - COALESCE(c.c_er, 0)) +
            (1 - COALESCE(c.c_rvr, 0)) +
            (1 - COALESCE(c.c_rvf, 0)) +
            (1 - COALESCE(c.c_rvs, 0))
        ) AS score_real_estate_proxy,

        s.sd_vitality_retail * (
            (1 - COALESCE(c.c_nvr, 0)) +
            (1 - COALESCE(c.c_pvr, 0)) +
            (1 - COALESCE(c.c_evr, 0)) +
            (1 - COALESCE(c.c_rvr, 0)) +
            (1 - COALESCE(c.c_vrvf, 0)) +
            (1 - COALESCE(c.c_vrvs, 0))
        ) AS score_vitality_retail,

        s.sd_vitality_food * (
            (1 - COALESCE(c.c_nvf, 0)) +
            (1 - COALESCE(c.c_pvf, 0)) +
            (1 - COALESCE(c.c_evf, 0)) +
            (1 - COALESCE(c.c_rvf, 0)) +
            (1 - COALESCE(c.c_vrvf, 0)) +
            (1 - COALESCE(c.c_vfvs, 0))
        ) AS score_vitality_food,

        s.sd_vitality_social * (
            (1 - COALESCE(c.c_nvs, 0)) +
            (1 - COALESCE(c.c_pvs, 0)) +
            (1 - COALESCE(c.c_evs, 0)) +
            (1 - COALESCE(c.c_rvs, 0)) +
            (1 - COALESCE(c.c_vrvs, 0)) +
            (1 - COALESCE(c.c_vfvs, 0))
        ) AS score_vitality_social
    FROM std s
    CROSS JOIN corrs c
),
weights AS (
    SELECT
        score_node_connectivity,
        score_place_population,
        score_place_employment,
        score_real_estate_proxy,
        score_vitality_retail,
        score_vitality_food,
        score_vitality_social,
        (
            score_node_connectivity +
            score_place_population +
            score_place_employment +
            score_real_estate_proxy +
            score_vitality_retail +
            score_vitality_food +
            score_vitality_social
        ) AS total_score
    FROM critic_scores
)
SELECT
    score_node_connectivity / NULLIF(total_score, 0) AS w_node_connectivity,
    score_place_population / NULLIF(total_score, 0) AS w_place_population,
    score_place_employment / NULLIF(total_score, 0) AS w_place_employment,
    score_real_estate_proxy / NULLIF(total_score, 0) AS w_real_estate_proxy,
    score_vitality_retail / NULLIF(total_score, 0) AS w_vitality_retail,
    score_vitality_food / NULLIF(total_score, 0) AS w_vitality_food,
    score_vitality_social / NULLIF(total_score, 0) AS w_vitality_social
FROM weights;
