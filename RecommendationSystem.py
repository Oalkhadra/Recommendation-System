"""
Methodology:
    My initial approach was to develop a hybrid model using Item-Based CF and XGBoost. After careful evaluation,
    I decided to focus on the XGBoost modeling approach. Two main components were critical to improving the model:

    1. Feature Engineering
        I explored numerous feature combinations and initially hoped to leverage NLP for analyzing texts from
        tip and review datasets. However, this approach led to very slow processing times with no significant
        improvement in RMSE. Ultimately, I used features from 5 of the 6 available files.

    2. Hyperparameter Tuning:
        I employed various approaches throughout the project, ranging from GridSearchCV to the Optuna library,
        to find the optimal set of parameters for the XGBRegressor.

    Other attempted approaches:
        - I also attempted to incorporate ensemble learning by adding a Neural Network to the modeling infrastructure. I
        tested multiple feed-forward networks, and tried a transformer learner as well. While I began to see promising
        results near the performance threshold, the Neural Network ultimately became too time-consuming to train and
        continued to under-perform.

        - Multiple NLP text analysis approaches were made, however, did not add much value to the model.

    Finally, an XGBRegressor model was used by itself, with more than 40 features extracted from the given datasets,
    tuned to avoid overfitting while ensuring accurate results.


Execution Time: Vocareum Terminal (380.08), Local Machine (984.86)
RMSE: 0.978
Error Distribution:
    {'>=0 and <1': 101994,
    '>=1 and <2': 33000,
    '>=2 and <3': 6238,
    '>=3 and <4': 812,
    '>=4': 0}
"""

from pyspark.sql import SparkSession
import numpy as np
from xgboost import XGBRegressor
import sys
import time
import json
import math
from datetime import datetime

def calculate_global_mean(business_features_rdd):
    """
    Calculate the global mean rating from business features.

    Args:
        business_features_rdd (RDD): RDD containing business features

    Returns:
        float: Global mean rating
    """
    # Extract all ratings and their corresponding review counts
    ratings_with_counts = business_features_rdd.map(lambda x: (x[1][1], x[1][0]))  # (stars, review_count)

    # Calculate weighted average
    total_weighted_rating = ratings_with_counts.map(lambda x: x[0] * x[1]).sum()
    total_reviews = ratings_with_counts.map(lambda x: x[1]).sum()

    # Return weighted mean, or default if no reviews
    return total_weighted_rating / max(total_reviews, 1) if total_reviews > 0 else 3.75


def extractBusinessFeatures(business_filepath):
    """
    Extracts relevant business features from a JSON file containing business data.

    Args:
        business_filepath (str): Path to the JSON file containing business data

    Returns:
        RDD: A PairRDD with business_id as key and a tuple of features as value:
            - business_id: Unique identifier for the business
            - Tuple containing:
                - review_count: Number of reviews for the business
                - stars: Average rating (1-5)
                - is_open: Business status (1=open, 0=closed)
                - attributes: Dictionary of business attributes
    """
    # Read the JSON file into an RDD and parse each line as JSON
    businessRDD = sc.textFile(business_filepath).map(lambda x: json.loads(x))

    # Extract key business features into a tuple format
    # Returns: (business_id, (review_count, average_stars, is_open, attributes))
    businessFeaturesRDD = businessRDD.map(lambda x: (x["business_id"],
                                                     (x["review_count"],  # Total number of reviews
                                                      x["stars"],  # Average rating (1-5 stars)
                                                      x["is_open"],  # Whether business is active (1) or closed (0)
                                                      x["attributes"])))  # Dictionary of business attributes

    return businessFeaturesRDD


def extractUserFeatures(users_filepath):
    """
    Extracts user features from a JSON file containing user data.

    Args:
        users_filepath (str): Path to the JSON file containing user data

    Returns:
        RDD: A PairRDD with user_id as key and a tuple of features as value:
            - user_id: Unique identifier for the user
            - Tuple containing:
                - average_stars: User's average rating given
                - review_count: Number of reviews written
                - fans: Number of fans
                - funny/cool/useful: Review feedback counts
                - elite_years: Number of years as elite user
                - days_yelping: Days since joining
                - various compliment counts
                - friend_count: Number of friends
    """
    # Read the JSON file into an RDD and parse each line as JSON
    usersRDD = sc.textFile(users_filepath).map(lambda x: json.loads(x))

    def calculate_days_yelping(yelping_since):
        """
        Calculate the number of days a user has been on Yelp.

        Args:
            yelping_since (str): Date string in format 'YYYY-MM-DD'

        Returns:
            int: Number of days since joining, or 0 if calculation fails
        """
        try:
            # Convert string date to datetime object
            start_date = datetime.strptime(yelping_since, "%Y-%m-%d")
            # Use fixed reference date for consistency
            reference_date = datetime(2024, 1, 1)
            # Calculate days between dates
            days_yelping = (reference_date - start_date).days
            return max(0, days_yelping)  # Ensure non-negative
        except:
            return 0  # Return 0 for any parsing errors

    # Extract key user features into a tuple format
    usersFeaturesRDD = usersRDD.map(lambda x: (x["user_id"],
                                               (x["average_stars"],  # Average rating given by user
                                                x["review_count"],  # Total number of reviews written
                                                x["fans"],  # Number of fans
                                                x["funny"],  # Number of funny votes received
                                                x["cool"],  # Number of cool votes received
                                                x["useful"],  # Number of useful votes received
                                                len(x["elite"]) if isinstance(x["elite"], list) else 0,
                                                # Years as elite user
                                                calculate_days_yelping(x["yelping_since"]),  # Days since joining
                                                x["compliment_funny"],  # Various compliment counts
                                                x["compliment_cool"],
                                                x["compliment_photos"],
                                                x["compliment_writer"],
                                                x["compliment_hot"],
                                                x["compliment_note"],
                                                len(x["friends"])
                                                    if isinstance(x["friends"], list) else 0)))  # Friend count

    return usersFeaturesRDD


def extractCheckinFeatures(checkin_filepath):
    """
    Extracts checkin features from a JSON file containing checkin data.

    Args:
        checkin_filepath (str): Path to the JSON file containing checkin data

    Returns:
        RDD: A PairRDD with business_id as key and a tuple of checkin metrics:
            - business_id: Unique identifier for the business
            - Tuple containing:
                - total_checkins: Total number of checkins
                - avg_checkins: Average number of checkins per time period
    """
    # Read the JSON file into an RDD and parse each line as JSON
    checkinRDD = sc.textFile(checkin_filepath).map(lambda x: json.loads(x))

    def getTotalCheckins(record):
        """
        Calculate total and average checkins for a business.

        Args:
            record (dict): Dictionary containing business_id and time data

        Returns:
            tuple: (business_id, (total_checkins, average_checkins))
        """
        business_id = record['business_id']
        time_dict = record['time']
        # Calculate total checkins across all time periods
        total_checkins = sum(time_dict.values())
        # Calculate average checkins per time period
        avg_checkins = sum(time_dict.values()) / len(time_dict.keys())

        return (business_id, (total_checkins, avg_checkins))

    # Transform checkin data into features
    checkinFeaturesRDD = checkinRDD.map(getTotalCheckins)

    return checkinFeaturesRDD


def extractPhotoFeatures(photo_filepath):
    """
    Extracts photo features from a JSON file containing photo data.

    Args:
        photo_filepath (str): Path to the JSON file containing photo data

    Returns:
        RDD: A PairRDD with business_id as key and photo count as value:
            - business_id: Unique identifier for the business
            - count: Number of photos associated with the business
    """
    # Read the JSON file into an RDD and parse each line as JSON
    photoRDD = sc.textFile(photo_filepath).map(lambda x: json.loads(x))

    # Group photos by business and count them
    photoFeaturesRDD = photoRDD.map(lambda x: (x["business_id"], x['photo_id'])) \
        .groupByKey() \
        .mapValues(len)  # Count number of photos per business

    return photoFeaturesRDD


def extractTipFeatures(tip_filepath):
    """
    Extracts tip features from a JSON file containing tip data.

    Args:
        tip_filepath (str): Path to the JSON file containing tip data

    Returns:
        RDD: A PairRDD with business_id as key and a tuple of tip metrics:
            - business_id: Unique identifier for the business
            - Tuple containing:
                - num_tips: Total number of tips
                - avg_likes_per_tip: Average likes per tip
                - total_likes: Total number of likes
                - avg_tip_length: Average tip text length
                - unique_users: Number of unique users who left tips
    """
    # Read the JSON file into an RDD and parse each line as JSON
    tipRDD = sc.textFile(tip_filepath).map(lambda x: json.loads(x))

    def process_tip(tip):
        """
        Process a single tip into initial statistics.

        Args:
            tip (dict): Dictionary containing tip data

        Returns:
            tuple: (business_id, dict of tip statistics)
        """
        business_id = tip['business_id']
        return (
            business_id,
            {
                'num_tips': 1,  # Count of tips
                'likes': tip['likes'],  # Number of likes
                'text_length': len(tip['text']),  # Length of tip text
                'users': {tip['user_id']},  # Set of unique users
            }
        )

    def combine_tip_stats(stats1, stats2):
        """
        Combine statistics from two sets of tips.

        Args:
            stats1, stats2 (dict): Dictionaries containing tip statistics

        Returns:
            dict: Combined statistics
        """
        return {
            'num_tips': stats1['num_tips'] + stats2['num_tips'],
            'likes': stats1['likes'] + stats2['likes'],
            'text_length': stats1['text_length'] + stats2['text_length'],
            'users': stats1['users'].union(stats2['users'])
        }

    # Process tips and combine statistics by business
    tipFeaturesRDD = tipRDD.map(process_tip).reduceByKey(combine_tip_stats)

    def calculate_final_features(business_stats):
        """
        Calculate final tip metrics from accumulated statistics.

        Args:
            business_stats (dict): Accumulated tip statistics for a business

        Returns:
            tuple: Final feature values
        """
        num_tips = business_stats['num_tips']
        return (
            num_tips,  # Total number of tips
            business_stats['likes'] / num_tips,  # Average likes per tip
            business_stats['likes'],  # Total number of likes
            business_stats['text_length'] / num_tips,  # Average tip length
            len(business_stats['users'])  # Number of unique users
        )

    return tipFeaturesRDD.mapValues(calculate_final_features)


def createFeatureVector(user_features_rdd, business_features_rdd, checkin_features_rdd,
                        photo_features_rdd, tip_features_rdd, data, mode):
    """
    Creates feature vectors for all user-business pairs by combining various feature RDDs.

    Args:
        user_features_rdd (RDD): RDD containing user features
        business_features_rdd (RDD): RDD containing business features
        checkin_features_rdd (RDD): RDD containing checkin features
        photo_features_rdd (RDD): RDD containing photo features
        tip_features_rdd (RDD): RDD containing tip features
        data (DataFrame): Input data containing user-business pairs
        mode (str): Either 'train' or 'test' to determine output format

    Returns:
        tuple: (feature_vectors_rdd, attribute_to_index)
            - feature_vectors_rdd: RDD containing ((user_id, business_id, rating), features)
            - attribute_to_index: Dictionary mapping business attributes to indices
    """
    # Convert RDDs to dictionaries for faster lookup
    user_features_map = user_features_rdd.collectAsMap()
    business_features_map = business_features_rdd.collectAsMap()
    checkin_features_map = checkin_features_rdd.collectAsMap()
    photo_features_map = photo_features_rdd.collectAsMap()
    tip_features_map = tip_features_rdd.collectAsMap()

    def collect_important_attributes(business_features_map):
        """
        Identifies business attributes that appear frequently enough to be meaningful.

        Args:
            business_features_map (dict): Dictionary of business features

        Returns:
            list: Sorted list of important attribute names
        """
        attribute_counts = {}

        # Count occurrences of each attribute
        for features in business_features_map.values():
            attributes = features[3]

            if attributes and attributes != 'None':
                # Handle string representation of attributes
                if isinstance(attributes, str):
                    try:
                        attributes = eval(attributes)
                    except:
                        continue

                if isinstance(attributes, dict):
                    for key, value in attributes.items():
                        attribute_counts[key] = attribute_counts.get(key, 0) + 1

        # Keep attributes that appear in at least 15% of businesses
        min_occurrences = len(business_features_map) * 0.15
        important_attributes = {
            attr for attr, count in attribute_counts.items()
            if count >= min_occurrences
        }

        return sorted(list(important_attributes))

    # Get list of important attributes and create index mapping
    attributes_list = collect_important_attributes(business_features_map)
    attribute_to_index = {attr: idx for idx, attr in enumerate(attributes_list)}
    total_attributes = len(attributes_list)

    # Broadcast feature dictionaries to all nodes
    user_features_broadcast = sc.broadcast(user_features_map)
    business_features_broadcast = sc.broadcast(business_features_map)
    checkin_features_broadcast = sc.broadcast(checkin_features_map)
    photo_features_broadcast = sc.broadcast(photo_features_map)
    tip_features_broadcast = sc.broadcast(tip_features_map)
    attribute_to_index_broadcast = sc.broadcast(attribute_to_index)

    def process_attributes(attributes, attribute_to_index_broadcast):
        """
        Processes business attributes into a numerical format.

        Args:
            attributes: Raw attributes data
            attribute_to_index_broadcast: Broadcast variable containing attribute index mapping

        Returns:
            list: List of (index, value) tuples for non-zero attributes
        """
        if not attributes or attributes == 'None' or attributes is None:
            return []

        # Handle string representation of attributes
        if isinstance(attributes, str):
            try:
                attributes = eval(attributes)
            except:
                return []

        if not isinstance(attributes, dict):
            return []

        # Convert attributes to numerical values
        attribute_features = []
        for attr, value in attributes.items():
            idx = attribute_to_index_broadcast.value.get(attr)
            if idx is not None:
                feature_value = 0.0

                # Convert different attribute types to numerical values
                if isinstance(value, bool):
                    feature_value = 1.0 if value else 0.0
                elif isinstance(value, str):
                    if value.lower() in ['true', 'yes', 'free', 'full_bar', 'beer_and_wine']:
                        feature_value = 1.0
                    elif value.lower() in ['false', 'no', 'none']:
                        feature_value = 0.0
                    elif value.isdigit():
                        feature_value = float(value)
                elif isinstance(value, (int, float)):
                    feature_value = float(value)

                if feature_value != 0.0:
                    attribute_features.append((idx, feature_value))

        return attribute_features

    def extract_features(row, user_features_broadcast, business_features_broadcast,
                         attribute_to_index_broadcast, checkin_features_broadcast,
                         photo_features_broadcast, tip_features_broadcast, total_attributes):
        """
        Extracts all features for a single user-business pair.

        Args:
            row: Input row containing user_id and business_id
            *_broadcast: Broadcast variables containing feature dictionaries
            total_attributes: Total number of business attributes

        Returns:
            tuple: ((user_id, business_id, rating), features list)
        """
        user_id = row['user_id']
        business_id = row['business_id']
        rating = float(row['stars']) if 'stars' in row else None

        features = []

        # Add user features with default values if user not found
        user_features = user_features_broadcast.value.get(user_id,
                                                          (GLOBAL_MEAN_RATING,
                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        # Basic user features
        features.extend([
            float(user_features[0]),  # average_stars
            float(user_features[1]),  # review_count
            float(user_features[2]),  # fans
            float(user_features[3]),  # funny votes
            float(user_features[4]),  # cool votes
            float(user_features[5]),  # useful votes
            float(user_features[6]),  # elite years
            float(user_features[7]),  # days_yelping
            float(user_features[8]),  # compliment_funny
            float(user_features[9]),  # compliment_cool
            float(user_features[10]),  # compliment_photos
            float(user_features[11]),  # compliment_writer
            float(user_features[12]),  # compliment_hot
            float(user_features[13]),  # compliment_note
            float(user_features[14]),  # number of friends

            # Derived user features
            math.log1p(float(user_features[7])),  # log of days_yelping
            float(user_features[1]) / max(float(user_features[7]), 1),  # reviews per day
            float(user_features[2]) / max(float(user_features[1]), 1),  # fans per review
            float(user_features[5]) / max(float(user_features[1]), 1)  # useful votes per review
        ])

        # Add business features with defaults
        business_features = business_features_broadcast.value.get(business_id, (0, GLOBAL_MEAN_RATING, 0, None))
        features.extend([
            float(business_features[0]),  # review_count
            float(business_features[1]),  # average stars
            float(business_features[2])  # is_open
        ])

        # Process business attributes
        attribute_features = np.zeros(total_attributes)
        if business_features[3]:
            for idx, value in process_attributes(business_features[3], attribute_to_index_broadcast):
                attribute_features[idx] = value
        features.extend(attribute_features)

        # Add checkin features and derived features
        checkin_features = checkin_features_broadcast.value.get(business_id, (0, 0))
        features.extend([
            float(checkin_features[0]),  # Total number of checkins
            float(checkin_features[1]),  # Avg number of checkins / day

            float(checkin_features[0]) / max(float(business_features[0]), 1), # Ratio of checkins and review counts per business
            float(checkin_features[0]) / max(float(business_features[1]), 1), # Ratio of checkins and avg stars per business
            float(checkin_features[1]) / max(float(business_features[0]), 1), # Ratio of avg checkins and review counts per business
            float(checkin_features[1]) / max(float(business_features[1]), 1)  # Ratio of avg checkins and avg stars per business

        ])

        # Add photo features
        photo_features = photo_features_broadcast.value.get(business_id, (0))
        features.extend([photo_features])  # number of photos

        # Add tip features
        tip_features = tip_features_broadcast.value.get(business_id, (0, 0, 0, 0, 0))
        features.extend([
            float(tip_features[0]),  # number of tips
            float(tip_features[1]),  # average likes per tip
            float(tip_features[2]),  # total likes
            float(tip_features[3]),  # average tip length
            float(tip_features[4])  # number of unique users

        ])

        # Return features with appropriate identifier based on mode
        if mode == 'train':
            return ((user_id, business_id, rating), features)
        elif mode == 'test':
            return ((user_id, business_id, None), features)

    # Create feature vectors for all rows in the input data
    feature_vectors_rdd = data.rdd.map(
        lambda row: extract_features(
            row,
            user_features_broadcast,
            business_features_broadcast,
            attribute_to_index_broadcast,
            checkin_features_broadcast,
            photo_features_broadcast,
            tip_features_broadcast,
            total_attributes
        )
    )

    return feature_vectors_rdd, attribute_to_index


def trainXGB(X_train, y_train):
    """
    Trains an XGBoost regression model on the given data.

    Args:
        X_train (np.array): Training feature matrix
        y_train (np.array): Training target values (ratings)

    Returns:
        XGBRegressor: Trained XGBoost model
    """
    # Initialize XGBoost model with optimized parameters
    model = XGBRegressor(
        max_depth=7,  # Maximum tree depth
        objective='reg:linear',  # Linear regression objective
        colsample_bytree=0.7,  # Fraction of features to use per tree
        subsample=0.7,  # Fraction of samples to use per tree
        min_child_weight=3,  # Minimum sum of instance weight in a child
        learning_rate=0.05,  # Learning rate
        n_estimators=500,  # Number of trees
        n_jobs=-1  # Use all available cores
    )

    print(f"Training the XGB model...")
    model.fit(X_train, y_train)

    return model


def predictXGB(model, X, ids):
    """
    Makes predictions using a trained XGBoost model.

    Args:
        model (XGBRegressor): Trained XGBoost model
        X (np.array): Feature matrix for prediction
        ids (np.array): Array of (user_id, business_id) pairs

    Returns:
        RDD: PairRDD containing (user_id, business_id, prediction)
    """
    print(f"Making the predictions...")

    # Generate raw predictions
    predictions = model.predict(X)

    # Clip predictions to valid rating range
    clipped_predictions = np.clip(predictions, 1, 5)

    # Combine IDs with predictions
    predictions_list = [
        (user_id, business_id, float(pred))
        for (user_id, business_id), pred in zip(ids, clipped_predictions)
    ]

    # Convert to RDD
    predictions_rdd = sc.parallelize(predictions_list)

    return predictions_rdd


def storeOutput(predicted_ratings, output_file_path):
    """
    Stores predicted ratings in a CSV file.

    Args:
        predicted_ratings (RDD): RDD containing (user_id, business_id, prediction) tuples
        output_file_path (str): Path to output CSV file
    """
    print(f"Storing output...")
    # Write predictions to CSV file
    with open(output_file_path, mode='w', encoding='utf-8') as f:
        # Write header
        f.write("user_id,business_id,prediction\n")
        # Write each prediction
        for user, business, prediction in predicted_ratings.collect():
            f.write(f"{user},{business},{prediction:.2f}\n")


if __name__ == "__main__":
    # Initialize Spark
    spark = (SparkSession.builder.appName('Competition').master('local[*]')
             .config("spark.driver.memory", "8g").config("spark.executor.memory", "8g")
             .getOrCreate())
    sc = spark.sparkContext.getOrCreate()

    # Start timer
    timeStart = time.time()

    # Read inputs
    folder_path = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Assign file paths and read train/test data
    trainFilepath = f"{folder_path}/yelp_train.csv"
    businessFilepath = f"{folder_path}/business.json"
    checkinFilepath = f"{folder_path}/checkin.json"
    photoFilepath = f"{folder_path}/photo.json"
    usersFilepath = f"{folder_path}/user.json"
    tipFilepath = f"{folder_path}/tip.json"
    reviewTrainFilepath = f"{folder_path}/review_train.json"
    trainData = spark.read.csv(trainFilepath, header=True)
    testData = spark.read.csv(test_filepath, header=True)

    # Extract features from data
    businessFeaturesRDD = extractBusinessFeatures(businessFilepath)
    userFeaturesRDD = extractUserFeatures(usersFilepath)
    checkinFeaturesRDD = extractCheckinFeatures(checkinFilepath)
    photoFeaturesRDD = extractPhotoFeatures(photoFilepath)
    tipFeaturesRDD = extractTipFeatures(tipFilepath)

    # Calculate and set global mean rating
    global GLOBAL_MEAN_RATING
    GLOBAL_MEAN_RATING = calculate_global_mean(businessFeaturesRDD)
    print(f"Global mean rating: {GLOBAL_MEAN_RATING}")

    # Define feature vector for train and test data
    feature_vectors_train, _ = createFeatureVector(
        userFeaturesRDD, businessFeaturesRDD, checkinFeaturesRDD, photoFeaturesRDD, tipFeaturesRDD,
        trainData, mode='train')

    feature_vectors_test, _ = createFeatureVector(
        userFeaturesRDD, businessFeaturesRDD, checkinFeaturesRDD, photoFeaturesRDD, tipFeaturesRDD,
        testData, mode='test')

    # Extract features and labels for train and test
    X_train = np.array(feature_vectors_train.map(lambda x: x[1]).collect())
    y_train = np.array(feature_vectors_train.map(lambda x: x[0][2]).collect())

    X = np.array(feature_vectors_test.map(lambda x: x[1]).collect())
    ids = np.array(feature_vectors_test.map(lambda x: (x[0][0], x[0][1])).collect())

    # Train XGB Model and predict ratings
    xgbModel = trainXGB(X_train, y_train)
    xgbPrediction = predictXGB(xgbModel, X, ids)

    # Store output
    storeOutput(xgbPrediction, output_filepath)
    print(f"Duration: {time.time() - timeStart}")