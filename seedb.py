from DB_Functions import DB_Functions
import matplotlib.pyplot as plt
import numpy as np
import CTE_fstrings
from scipy.special import kl_div
import os
import time
import warnings
warnings.filterwarnings("ignore")

class SeeDB:
    def __init__( self , db_name , user , measure_attributes , dimension_attributes , table ):
        self.db_handler = DB_Functions()
        self.database_connection = self.db_handler.connect_database( db_name , user )
        self.num_rows = self.db_handler.fetch_data( self.database_connection ,
        ( 'select count(*) from ' + table ))[ 0 ][ 0 ]
        self.measure_attributes = measure_attributes
        self.dimension_attributes = dimension_attributes
        self.table = table
        self.aggregate_fns = [ 'max' , 'sum' , 'count' ,  'avg' , 'min' ]

    def compute_kl_div( self , prob_dist_1 , prob_dist_2 ):
        '''
        Compute KL divergence between given probability distributions
        '''
        epsilon = 1e-15
        prob_dist_1 = prob_dist_1/( np.sum( prob_dist_1 ) + epsilon )
        prob_dist_2 = prob_dist_2/( np.sum( prob_dist_2 ) + epsilon )
        prob_dist_1 = np.clip( prob_dist_1 , epsilon , None )
        prob_dist_2 = np.clip( prob_dist_2 , epsilon , None )
        kl_divg = np.sum( kl_div( prob_dist_1 , prob_dist_2 ) )
        return kl_divg

    def populate_potential_view_params( self ):
        '''
        Populate dimension , measure and aggregate function combinations for potential views
        '''
        potential_views = dict()
        for dimension_attribute in self.dimension_attributes:
            for measure_attribute in self.measure_attributes:
                for aggregate_fn in self.aggregate_fns:
                    if dimension_attribute not in potential_views:
                        potential_views[ dimension_attribute ] = dict()
                    if measure_attribute not in potential_views[ dimension_attribute ]:
                        potential_views[ dimension_attribute ][ measure_attribute ] = set()
                    potential_views[ dimension_attribute ][ measure_attribute ].add( aggregate_fn )
        return potential_views

    def get_suggested_views( self , potential_views ):
        '''
        Get the result of suggested views
        '''
        suggested_views = []
        for dimension_attribute in potential_views:
            for measure_attribute in potential_views[ dimension_attribute ]:
                for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                    suggested_views.append( ( dimension_attribute , measure_attribute , aggregate_fn ) )
        return suggested_views

    def get_combined_selection_query( self , potential_views , dimension_attribute , user_dataset_condition , reference_dataset_condition ):
        '''
        Get the selection query that is used to get both the user and the reference data
        '''
        user_selection = ''
        reference_selection = ''
        for measure_attribute in potential_views[ dimension_attribute ]:
            for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                user_fn = CTE_fstrings.user_fn_cte_fstring.format( fn = aggregate_fn , 
                    condition = user_dataset_condition , 
                    measure_attribute = measure_attribute
                )

                reference_fn = CTE_fstrings.reference_fn_cte_fstring.format(
                    fn = aggregate_fn ,
                    condition = reference_dataset_condition ,
                    measure_attribute = measure_attribute
                )

                user_selection += '{}, '.format( user_fn )
                reference_selection += '{}, '.format( reference_fn )

        combined_selection = user_selection + reference_selection[ : -2 ]
        return combined_selection

    def delete_pruned_views( self , potential_views , distance_index_mappings_view , pruned_view_indexes , view_distances  ):
        '''
        Delete the pruned views from the potential views
        '''
        pruned_views = [ distance_index_mappings_view[ id ] for id in pruned_view_indexes ]
        for dimension_attribute, measure_attribute, aggregate_fn in pruned_views:
            potential_views[ dimension_attribute ][ measure_attribute ].remove( aggregate_fn )
            if len( potential_views[ dimension_attribute ][ measure_attribute ] ) == 0:
                del potential_views[ dimension_attribute ][ measure_attribute ]
                if len( potential_views[ dimension_attribute ] ) == 0:
                    del potential_views[ dimension_attribute ]
        return( potential_views )

    def split_combined_df( self , combine_df ):
        '''
        Split combined dataframe into those corresponding to user and reference
        '''
        r = self.filter_df_keyword( combine_df , 'reference' )
        r = np.array( r ) 
        q = self.filter_df_keyword( combine_df , 'user' )
        q = np.array( q )
        return( [ r , q  ] )
    
    def suggest_views_sharing( self , user_dataset_condition , reference_dataset_condition , total_phases = 10 ):
        '''
        Implement view recommendation logic using only sharing optimization
        '''
        dist = self.compute_kl_div 
        potential_views = self.populate_potential_view_params()
        current_phase =0
        for start, end in self.generate_indices_phase(total_phases):
            current_phase += 1
            current_view = -1
            view_distances = []
            distance_index_mappings_view = dict()
            for dimension_attribute in potential_views:
                attributes_cte = CTE_fstrings.attributes_cte_fstring.format( attribute = dimension_attribute ,
                table = self.table )
                all_selections = self.get_combined_selection_query( potential_views , dimension_attribute , user_dataset_condition , reference_dataset_condition )
                combined_query = CTE_fstrings.combined_query_cte_fstring.format(
                    attributes_cte = attributes_cte ,
                    all_selections = all_selections ,
                    table = self.table ,
                    attribute = dimension_attribute ,
                    start = start ,
                    end = end
                )
                combine_df = self.db_handler.fetch_data( self.database_connection , combined_query , return_df = True )
                [ r , q ] = self.split_combined_df( combine_df )
                current_col = -1
                for measure_attribute in potential_views[ dimension_attribute ]:
                    for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                        current_view += 1
                        current_col += 1
                        d = dist( q[ : , current_col ] , r[ : , current_col ] )
                        view_distances.append( d )
                        distance_index_mappings_view[ current_view ] = ( dimension_attribute , measure_attribute , aggregate_fn )
        sorted_views = sorted(
            ((distance_index_mappings_view[i], divergence) for i, divergence in enumerate(view_distances)),
            key=lambda x: x[1],  
            reverse=True
        )
        top_k_views = sorted_views[:5]
        recommended_views = [view for view, _ in top_k_views]

        return recommended_views
            
    def suggest_views_pruning( self , user_dataset_condition , reference_dataset_condition , total_phases = 10 ):
        '''
        Implement view recommendation logic using only pruning optimization
        '''
        dist = self.compute_kl_div
        potential_views = self.populate_potential_view_params()
        current_phase =0
        for start, end in self.generate_indices_phase(total_phases):
            current_phase += 1
            current_view = -1
            view_distances = []
            distance_index_mappings_view = dict()
            for dimension_attribute in potential_views:
                for measure in potential_views[dimension_attribute]:
                    for func in potential_views[dimension_attribute][measure]:
                        user_sql = CTE_fstrings.prune_user_query.format(attribute = dimension_attribute, table= self.table,measure = measure, func = func, start= start, end = end,query_dataset_cond=user_dataset_condition)
                        reference_sql = CTE_fstrings.prune_reference_query.format(attribute = dimension_attribute, table= self.table,measure = measure, func = func, start= start, end = end, reference_dataset_cond = reference_dataset_condition)
                        user_df = self.db_handler.fetch_data(self.database_connection, user_sql, return_df = True)
                        reference_df = self.db_handler.fetch_data(self.database_connection, reference_sql, return_df = True)
                u = np.array(user_df['query_result'].fillna(0)) 
                r = np.array(reference_df['reference_result'].fillna(0))  
                for measure in potential_views[dimension_attribute]:
                    for func in potential_views[dimension_attribute][measure]:
                        current_view += 1     
                        d = dist(u, r)
                        view_distances.append(d)
                        distance_index_mappings_view[current_view] = (dimension_attribute, measure, func)
            pruned_view_indexes = self.implement_pruning(view_distances, current_phase)
            potential_views = self.delete_pruned_views(potential_views, distance_index_mappings_view, pruned_view_indexes, view_distances)
        suggested_views = self.get_suggested_views(potential_views)
        return suggested_views


    def suggest_views_combined( self , user_dataset_condition , reference_dataset_condition , total_phases = 10 ):
        '''
        Implement view recommendation logic using combined( sharing and pruning ) optimization
        '''
        dist_fn_pruning = self.compute_kl_div 
        potential_views = self.populate_potential_view_params()
        current_phase = 0
        for start, end in self.generate_indices_phase( total_phases ):
            current_phase += 1
            current_view = -1
            view_distances = []
            distance_index_mappings_view = dict()
            for dimension_attribute in potential_views:
                attributes_cte = CTE_fstrings.attributes_cte_fstring.format( attribute = dimension_attribute ,
                table = self.table )
                all_selections = self.get_combined_selection_query( potential_views , dimension_attribute , user_dataset_condition , reference_dataset_condition )
                combined_query = CTE_fstrings.combined_query_cte_fstring.format(
                    attributes_cte = attributes_cte ,
                    all_selections = all_selections ,
                    table = self.table ,
                    attribute = dimension_attribute ,
                    start = start ,
                    end = end
                )
                combine_df = self.db_handler.fetch_data( self.database_connection , combined_query , return_df = True )
                [ r , q ] = self.split_combined_df( combine_df )
                current_col = -1
                for measure_attribute in potential_views[ dimension_attribute ]:
                    for aggregate_fn in potential_views[ dimension_attribute ][ measure_attribute ]:
                        current_view += 1
                        current_col += 1
                        d = dist_fn_pruning( q[ : , current_col ] , r[ : , current_col ] )
                        view_distances.append( d )
                        distance_index_mappings_view[ current_view ] = ( dimension_attribute , measure_attribute , aggregate_fn )
            pruned_view_indexes = self.implement_pruning( view_distances , current_phase )
            potential_views = self.delete_pruned_views( potential_views , distance_index_mappings_view , pruned_view_indexes ,
            view_distances  )
        suggested_views = self.get_suggested_views( potential_views )
        return suggested_views
    
    def construct_combined_view_query( self , selections_query , table , dimension_attribute , start , end ):
        '''
        Construct the combined view query
        '''
        return CTE_fstrings.construct_combined_view_query_fstring.format(
            attribute = dimension_attribute ,
            selections_query = selections_query ,
            table = table ,
            start = start ,
            end = end
        )

    def filter_df_keyword( self , df , keyword ):
        '''
        Filter the dataframe using keyword passed
        '''
        filtered_columns =  [ col for col in df.columns if keyword in col ] 
        filtered_columns = list( dict.fromkeys( filtered_columns ) )
        return df[ filtered_columns ]

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
            self.table , user_dataset_condition , dimension_attribute ,  0 , self.num_rows )
            reference_dataset_query = self.construct_view_query( select_str ,
            self.table , reference_dataset_condition , dimension_attribute , 0 , self.num_rows )
            user_data = np.array( self.db_handler.fetch_data( self.database_connection , user_dataset_query ))
            reference_data = np.array( self.db_handler.fetch_data( self.database_connection , reference_dataset_query ))
            user_measure_aggregates = user_data[ : , 1 ].astype( float )
           
            reference_measure_aggregates = reference_data[ : , 1 ].astype( float )
            kl_divergence = self.compute_kl_div(user_measure_aggregates,reference_measure_aggregates)
            divergences.append((view, kl_divergence, user_data, reference_measure_aggregates, user_measure_aggregates))
        divergences.sort(key=lambda x: x[1], reverse=True)


        for view, divergence, user_data, reference_measure_aggregates, user_measure_aggregates in divergences:
            dimension_attribute, measure_attribute, aggregate_fn = view
            user_measure_labels = user_data[ : , 0 ] 
            print("View:", view, ", Divergence:", divergence)
            self.generate_plot(user_data, user_measure_aggregates, reference_measure_aggregates, labels, dimension_attribute, aggregate_fn, measure_attribute, user_measure_labels, folder_path)


    def generate_indices_phase( self , total_phases = 10 ):
        '''
        Generate indices batches
        '''
        batch_size = self.num_rows // total_phases
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


if __name__ == '__main__':
    start_time = time.time() 

    seedb = SeeDB( 'seedb' , 'postgres' , [ 'fnlwgt' , 'age' , 'capital_gain' , 'capital_loss' , 'hours_per_week' ] ,
    [ 'workclass' , 'education' , 'occupation' , 'relationship' , 'race' , 'native_country' , 'salary', 'sex' ] , 'census' )
    labels = [ 'Married' , 'Unmarried' ]
    user_dataset = "marital_status in (' Married-civ-spouse', ' Married-spouse-absent', ' Married-AF-spouse')"
    reference_dataset = "marital_status in (' Divorced', ' Never-married', ' Separated', ' Widowed')"

    print("Pruning+Sharing Optimization")
    start_time = time.time() 
    #Combined sharing and pruning
    views = seedb.suggest_views_combined( user_dataset , reference_dataset , 5 )
    
    seedb.create_visualizations( views , user_dataset , reference_dataset ,  [ 'Married' , 'Unmarried' ] ,
    folder_path = 'visualizations/combined/' )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time (Combined): {elapsed_time:.4f} seconds")

    print("Sharing Only Optimization")
    start_time = time.time() 
    # sharing 
    views = seedb.suggest_views_sharing( user_dataset , reference_dataset , 5 )
    
    seedb.create_visualizations( views , user_dataset , reference_dataset ,  [ 'Married' , 'Unmarried' ] ,
    folder_path = 'visualizations/sharing/' )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time (Sharing): {elapsed_time:.4f} seconds")

    # pruning 
    print("Pruning Only Optimization")
    views = seedb.suggest_views_pruning( user_dataset , reference_dataset , 5 )
    
    seedb.create_visualizations( views , user_dataset , reference_dataset ,  [ 'Married' , 'Unmarried' ] ,
    folder_path = 'visualizations/pruning/' )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time (Pruning): {elapsed_time:.4f} seconds")



    

