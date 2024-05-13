# SEEDB Project Repository

This repository contains all the necessary scripts and SQL files for the implementation of the SEEDB project, which aims to optimize and visualize data from the U.S. Census dataset for insightful analysis based on marital status differences.

## Repository Structure

- **visualizations/**: Contains generated visualization files.
- **CTE_fstrings.py**: Utility scripts for Common Table Expressions using f-strings.
- **DB_Functions.py**: Database functions for various operations within SEEDB.
- **combine_multiple_groupbys.py**: Script for combining multiple GROUP BY operations in queries.
- **database.sql**: SQL scripts for database schema and initial data loading.
- **preprocess.py**: Script for preprocessing the input data.
- **seedb.py**: Main script for running the SEEDB implementation.

## Getting Started

To get started with the SEEDB project implementation, follow the steps outlined below:

### Prerequisites

Ensure you have Python and PostgreSQL installed on your system. The scripts are tested with Python 3.8 and PostgreSQL 12. Adjustments might be necessary for other versions.

### Running the Scripts

1. **Database Setup**:
   - Run the `database.sql` script to set up your database schema and load the initial data:
     ```
     psql -U username -d your_database -a -f database.sql
     ```

2. **Data Preprocessing**:
   - Execute the `preprocess.py` script to preprocess the input data:
     ```
     python preprocess.py
     ```

3. **SEEDB Implementation**:
   - To run the main SEEDB implementation and generate visualizations, execute:
     ```
     python seedb.py
     ```

4. **Additional Group By Optimizations**:
   - For additional optimizations involving multiple GROUP BY queries, run the `combine_multiple_groupbys.py` script:
     ```
     python combine_multiple_groupbys.py
     ```



