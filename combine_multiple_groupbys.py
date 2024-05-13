import numpy as np
import math
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
from scipy.special import kl_div
import time
from DB_Functions import DB_Functions
import CTE_fstrings
import os
import warnings
warnings.filterwarnings("ignore")


class SeeDB(object):
    def __init__(self, db_name, db_user, metrics, categories, data_table):
        connection_data = establish_connection(db_name, db_user)
        self.connection, self.record_count = (connection_data['conn'], connection_data['count'])
        self.metrics = metrics
        self.categories = categories
        self.data_table = data_table
        self.aggregations = ['avg', 'min', 'max', 'sum', 'count']
    
        self.db_handler = DB_Functions()

    def generate_indices_phase( self , total_phases = 10 ):
        '''
        Generate indices batches
        '''
        batch_size = self.record_count // total_phases
        return [ ( batch_size * i, batch_size * ( i + 1 ) ) for i in range( total_phases ) ]
    def calc_confidence_error( self , current_phase, total_phases = 10 ):
        '''
        Calculate conidence error based on confidence interval
        '''
        delta = 0.05
        confidence_error = np.sqrt(( 1.-(current_phase/total_phases) )*(( 2*np.log(np.log(current_phase+1)) )+
        ( np.log(np.pi**2/(3*delta)) ))*( 0.5*1/(current_phase+1) ))
        return confidence_error
    
    def implement_pruning( self , kl_divergences , current_phase , total_phases = 10 , delta = 0.05 , top_k = 5 ):
        '''
        Implement pruning
        '''
        if current_phase == 1:
            return []
        kl_divergences = np.array(kl_divergences)

        sorted_indices = np.argsort(kl_divergences)[::-1]
        sorted_kl_divergences = kl_divergences[sorted_indices]

        if current_phase == total_phases:
            return sorted_indices[top_k:]

        confidence_error = self.calc_confidence_error(current_phase, total_phases)
        threshold_kl = sorted_kl_divergences[top_k - 1] - confidence_error
        if len(sorted_kl_divergences) > top_k:
            prune_start_index = np.argmax(sorted_kl_divergences[top_k:] + confidence_error < threshold_kl) + top_k
            return sorted_indices[prune_start_index:] if prune_start_index < len(sorted_kl_divergences) else []

        return []

    def group_optimization(self, attribute_info, max_memory):
        log_sizes = {attr: math.log(size) for attr, size in attribute_info.items()}
        ordered_attributes = sorted(log_sizes, key=log_sizes.get, reverse=True)
        bins = []
        for attr in ordered_attributes:
            placed = False
            for bin in bins:
                if sum(log_sizes[a] for a in bin) + log_sizes[attr] <= max_memory:
                    bin.append(attr)
                    placed = True
                    break
            if not placed:
                bins.append([attr])
        return bins

    def keyword_filtered_columns(self, dataframe, keyword):
        filtered_columns = [col for col in dataframe.columns if keyword in col]
        filtered_columns = list(dict.fromkeys(filtered_columns))
        return dataframe[filtered_columns]

    

    def prune_views(self, kl_divergence_values, current_phase, total_phases=10, delta=0.05, max_views=5):
        kl_values = np.array(kl_divergence_values)
        if current_phase == 1:
            return []

        total_values = len(kl_values)

        sorted_kl = np.sort(kl_values)[::-1]
        sorted_indices = np.argsort(kl_values)[::-1]

        if current_phase == total_phases:
            return sorted_indices[max_views:]

        a_value = 1 - (current_phase / total_phases)
        b_value = 2 * np.log(np.log(current_phase + 1))
        c_value = np.log(np.pi ** 2 / (3 * delta))
        d_value = 0.5 * 1 / (current_phase + 1)
        error_margin = np.sqrt(a_value * (b_value + c_value) * d_value)

        min_kl_value = sorted_kl[max_views - 1] - error_margin
        for i in range(max_views, total_values):
            if sorted_kl[i] + error_margin < min_kl_value:
                return sorted_indices[i:]
        return []

    def generate_suggestions(self, user_condition, reference_condition, total_phases=10):
        attribute_info = {'workclass': 9, 'education': 16, 'occupation': 15, 'relationship': 6, 'race': 5, 'native_country': 42, 'salary': 2, 'sex': 2}
        memory_limit = 5
        grouped_attributes = self.group_optimization(attribute_info, memory_limit)
        divergence = compute_kl_divergence
        view_candidates = {}

        view_candidates = {
            tuple(attr): {metric: self.aggregations[:] for metric in self.metrics}
            for attr in grouped_attributes
        }

        phase_counter = 0
        for start, end in self.generate_indices_phase(total_phases):
            phase_counter += 1

            view_index = -1
            divergence_values = []
            index_to_view_map = {}
            for group in grouped_attributes:
                group_str = ', '.join(group)
                cte_attr = ',\n'.join([f"{attr}_cte AS (SELECT DISTINCT {attr} FROM {self.data_table})" for attr in group])
                cross_join = ' CROSS JOIN '.join([f"{attr}_cte" for attr in group])

                combined_cte = f"attrs AS (SELECT {group_str} FROM {cross_join})"

                selections = []
                for metric in self.metrics:
                    for agg in self.aggregations:
                        user_select = f"COALESCE({agg}(CASE WHEN {user_condition} THEN {metric} END), 0) AS user_{agg}_{metric}"
                        ref_select = f"COALESCE({agg}(CASE WHEN {reference_condition} THEN {metric} END), 0) AS ref_{agg}_{metric}"
                        selections.append(user_select)
                        selections.append(ref_select)

                selection_str = ', '.join(selections)
                cte_full = f"WITH {cte_attr},\n{combined_cte}"

                query = f"""
                {cte_full}
                SELECT {', '.join(['attrs.' + attr for attr in group])}, {selection_str}
                FROM attrs
                LEFT JOIN {self.data_table} ON {' AND '.join([f'attrs.{attr} = {self.data_table}.{attr}' for attr in group])}
                AND id >= {start} AND id < {end}
                GROUP BY {', '.join(['attrs.' + attr for attr in group])}
                ORDER BY {', '.join(['attrs.' + attr for attr in group])};
                """
                combined_df = self.db_handler.fetch_data(self.connection, query, return_df = True)

                ref_array = np.array(self.keyword_filtered_columns(combined_df, 'ref'))
                user_array = np.array(self.keyword_filtered_columns(combined_df, 'user'))

                col_index = -1
                if tuple(group) in view_candidates:
                    for metric in view_candidates[tuple(group)]:
                        for agg in view_candidates[tuple(group)][metric]:
                            view_index += 1
                            col_index += 1
                            divergence_val = divergence(user_array[:, col_index], ref_array[:, col_index])
                            divergence_values.append(divergence_val)
                            index_to_view_map[view_index] = (tuple(group), metric, agg)

            views_to_remove = self.implement_pruning(divergence_values, phase_counter)
            views_to_prune = [index_to_view_map[idx] for idx in views_to_remove]

            try:
                for group, metric, agg in views_to_prune:
                    view_candidates[group][metric].remove(agg)

                    if not view_candidates[group][metric]:
                        del view_candidates[group][metric]
                        if not view_candidates[group]:
                            del view_candidates[group]
            except:
                print("Error occurred during pruning.")

        suggested_views = []
        for group in view_candidates:
            for metric in view_candidates[group]:
                for agg in view_candidates[group][metric]:
                    suggested_views.append((group, metric, agg))

        return suggested_views

    

    def construct_view_query( self , selections_query , table , condition , dimension_attribute , start , end ):
        '''
        Construct the view query
        '''
        view_query = CTE_fstrings.construct_view_query_fstring.format(dimension_attribute, table, selections_query, condition, start, end)
        return view_query.strip()

    def generate_plot( self , user_data , user_measure_aggregates , reference_measure_aggregates , labels ,
    dimension_attribute , aggregate_fn , measure_attribute , user_measure_labels , folder_path ):
        '''
        Generate plots based on parameters passed
        '''
        plt.figure()
        num_groups = len( user_measure_aggregates ) 
        index = np.arange( num_groups )
        bar_width = 0.35
        opacity = 0.8

        plt.bar(index, user_measure_aggregates, bar_width, alpha=opacity, color='b', label=labels[0])
        plt.bar(index + bar_width, reference_measure_aggregates, bar_width, alpha=opacity, color='g', label=labels[1])

        plt.xlabel(dimension_attribute)
        plt.ylabel('{}({})'.format(aggregate_fn, measure_attribute))
        if len(user_measure_labels) == num_groups:
            plt.xticks(index + bar_width / 2, user_measure_labels, rotation=90)
        else:
            print("Warning: The number of labels does not match the number of x-ticks.")

        plt.legend()
        plt.tight_layout()

        plot_filename = '{}_{}_{}.png'.format(dimension_attribute, measure_attribute, aggregate_fn)
        plt.savefig(os.path.join(folder_path, plot_filename), dpi=300)
        plt.close()

    def create_visualizations( self , views , user_dataset_condition , reference_dataset_condition ,labels  ,
    folder_path = 'visualizations/' ):
        '''
        Generate the visualizations given views
        '''
        divergences=[]
        for view in views:
            dimension_attribute, measure_attribute, aggregate_fn = view
            select_str = CTE_fstrings.create_visualizations_fstring.format( aggregate_fn , measure_attribute )
            user_dataset_query = self.construct_view_query( select_str ,
            self.data_table , user_dataset_condition , dimension_attribute ,  0 , self.record_count )
            reference_dataset_query = self.construct_view_query( select_str ,
            self.data_table , reference_dataset_condition , dimension_attribute , 0 , self.record_count )
            user_data = np.array( self.db_handler.fetch_data( self.connection , user_dataset_query ))
            reference_data = np.array( self.db_handler.fetch_data( self.connection , reference_dataset_query ))
            user_measure_aggregates = user_data[ : , 1 ].astype( float )
           
            reference_measure_aggregates = reference_data[ : , 1 ].astype( float )
            kl_divergence = compute_kl_divergence(user_measure_aggregates,reference_measure_aggregates)
            divergences.append((view, kl_divergence, user_data, reference_measure_aggregates, user_measure_aggregates))
        divergences.sort(key=lambda x: x[1], reverse=True)


        for view, divergence, user_data, reference_measure_aggregates, user_measure_aggregates in divergences:
            dimension_attribute, measure_attribute, aggregate_fn = view
            user_measure_labels = user_data[ : , 0 ] 
            print("View:", view, ", Divergence:", divergence)
            self.generate_plot(user_data, user_measure_aggregates, reference_measure_aggregates, labels, dimension_attribute, aggregate_fn, measure_attribute, user_measure_labels, folder_path)


def establish_connection(db_name, user):
    conn = psycopg2.connect(database=db_name, user=user, password="Seafoam@6613", host="localhost", port="5432")
    table = 'census'
    return {'conn': conn,
            'count': int(retrieve_records(conn, ('select count(*) from ' + table))[0][0])}


def retrieve_records(conn, query, as_float=False):
    cursor = conn.cursor()
    cursor.execute(query)
    records = cursor.fetchall()
    if as_float:
        records = np.array(records).astype(float)
    return records


def retrieve_data(conn, query, params=None):
    return pd.read_sql_query(query, conn, params=params)


def compute_kl_divergence(array1, array2):
    epsilon = 1e-15
    array1 = array1 / (np.sum(array1) + epsilon)
    array2 = array2 / (np.sum(array2) + epsilon)
    array1 = np.clip(array1, epsilon, None)
    array2 = np.clip(array2, epsilon, None)
    kl_value = np.sum(kl_div(array1, array2))
    return kl_value

def flatten_groups(tuples_of_data):
    flat_list = []
    for group, field, operation in tuples_of_data:
        for element in group:
            flat_list.append((element, field, operation))
    return flat_list


def main(database_name):
    db_user = 'postgres'
    metrics = ['age', 'capital_gain', 'capital_loss', 'hours_per_week', 'fnlwgt']
    categories = ['workclass', 'education', 'occupation', 'relationship',
                  'race', 'native_country', 'salary', 'sex']
    census_table = 'census'

    analyzer = SeeDB(database_name, db_user, metrics, categories, census_table)

    user_condition = "marital_status in (' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse')"
    reference_condition = "marital_status in (' Divorced', ' Never-married', ' Separated', ' Widowed')"

    suggested_views = analyzer.generate_suggestions(user_condition, reference_condition)
    print('Suggested views ', suggested_views)

    structured_data = flatten_groups(suggested_views)

    comparison_labels = ['Married', 'Unmarried']

    analyzer.create_visualizations(structured_data, user_condition, reference_condition, comparison_labels, folder_path='visualizations/multiple_groupbys/')


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()    

    db_name = 'seedb'
    main(db_name)

    # Record the end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
