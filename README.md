# Yelp Recommendation System

## Overview
An advanced recommendation system built using Apache Spark (RDD) and XGBoost that predicts user ratings for businesses on Yelp. The system employs extensive feature engineering and machine learning techniques to achieve highly accurate predictions with an RMSE of 0.978.

## Methodology
The system uses a hybrid approach focusing on:

1. **Feature Engineering**: Comprehensive feature extraction from multiple data sources:
   - User metadata and behavior patterns
   - Business attributes and characteristics
   - Check-in patterns
   - Photo engagement metrics
   - Tip and review analysis
   - Derived features and ratios

2. **XGBoost Model**:
   - Optimized hyperparameters for robust predictions
   - Parameters tuned to prevent overfitting while maintaining accuracy
   - Implementation leverages over 40 engineered features

## Performance Metrics
- **RMSE**: 0.978
- **Error Distribution**:
  - >=0 and <1: 101,994 predictions
  - >=1 and <2: 33,000 predictions
  - >=2 and <3: 6,238 predictions
  - >=3 and <4: 812 predictions
  - >=4: 0 predictions
- **Execution Time**: 
  - Vocareum: 380.08s
  - Local Machine: 984.86s

## Technical Implementation

### Dependencies
- PySpark
- XGBoost
- NumPy
- JSON
- DateTime

### Dataset Components
- `yelp_train.csv`: Training dataset
- `business.json`: Business metadata
- `checkin.json`: User check-in data
- `photo.json`: Photo metadata
- `user.json`: User information
- `tip.json`: User tips/short reviews
- `review_train.json`: Review data for training pairs

### Key Features
1. Business Features:
   - Review counts
   - Average ratings
   - Business status
   - Location attributes

2. User Features:
   - Historical ratings
   - Review counts
   - User engagement metrics
   - Social indicators

3. Interaction Features:
   - Check-in patterns
   - Photo engagement
   - Tip statistics
   - Combined user-business metrics

## Usage

### Input Format
```bash
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
```

### Parameters
- `folder_path`: Path to dataset folder
- `test_file_name`: Name of test file (e.g., yelp_val.csv)
- `output_file_name`: Path for prediction results

### Output Format
CSV file with columns:
- user_id
- business_id
- prediction

## Development Notes
- Extensive use of Spark RDD for data processing
- Optimized data structures for efficient memory usage
- Careful handling of missing values and edge cases
- Robust error handling and input validation

## Project Requirements
- Python 3.6
- Spark 3.1.2
- All feature engineering must be done using Spark RDD
- Execution time must not exceed 25 minutes

## Optimization Techniques
1. Efficient data structure usage
2. Memory-optimized feature extraction
3. Parallel processing with Spark RDD
4. Careful selection of XGBoost parameters
5. Optimized data preprocessing pipeline

## Future Improvements
- Integration of neural networks for hybrid modeling
- Advanced text analysis of reviews
- Additional feature engineering possibilities
- Model ensemble techniques
- Performance optimization for larger datasets

## Author
Omar Alkhadra
