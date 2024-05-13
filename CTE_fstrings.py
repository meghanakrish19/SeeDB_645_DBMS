attributes_cte_fstring = "WITH attrs AS (SELECT DISTINCT {attribute} FROM {table})"
user_fn_cte_fstring = "COALESCE({fn}(CASE WHEN {condition} THEN {measure_attribute} END), 0) AS user_{fn}_{measure_attribute}"
reference_fn_cte_fstring = 'COALESCE({fn}(CASE WHEN {condition} THEN {measure_attribute} END), 0) AS reference_{fn}_{measure_attribute}'
combined_query_cte_fstring = """
{attributes_cte}
SELECT {all_selections}
FROM attrs a
LEFT JOIN {table} ON a.{attribute} = {table}.{attribute}
AND id >= {start} AND id < {end}
GROUP BY a.{attribute}
ORDER BY a.{attribute};
"""
construct_combined_view_query_fstring = """
SELECT {attribute}, {selections_query}
FROM {table}
WHERE {attribute} IS NOT NULL AND id >= {start} AND id < {end}
GROUP BY {attribute}
ORDER BY {attribute};
"""
construct_view_query_fstring = """
        WITH attrs AS (
            SELECT DISTINCT({0}) AS __atr__
            FROM {1}
        )
        SELECT {2}
        FROM attrs
        LEFT OUTER JOIN {1} ON __atr__ = {0}
            AND {3}
            AND id >= {4}
            AND id < {5}
        GROUP BY __atr__
        ORDER BY __atr__
        """

prune_user_query = """with t1 as (SELECT {attribute} as h1, {func}({measure}) AS q_result
                                    FROM {table}
                                    WHERE {query_dataset_cond} AND id >= {start} AND id < {end}
                                    GROUP BY {attribute}
                                    ORDER BY {attribute}),
                                    t2 as (SELECT DISTINCT {attribute} as h2 FROM census )
                                    select h2,COALESCE( q_result,0) as query_result from t1 right join t2 on t1.h1 = t2.h2 ; """
prune_reference_query ="""with t1 as (SELECT {attribute} as h1, {func}({measure}) AS r_result
                                    FROM {table}
                                    WHERE {reference_dataset_cond} AND id >= {start} AND id < {end}
                                    GROUP BY {attribute}
                                    ORDER BY {attribute}),
                                    t2 as (SELECT DISTINCT {attribute} as h2 FROM census )
                                    select h2,COALESCE( r_result,0) as reference_result from t1 right join t2 on t1.h1 = t2.h2"""

create_visualizations_fstring = "__atr__, COALESCE({}({}), 0)"
